#include "snakelib_estimation/estimator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <optional>

#include "Eigen/Dense"
#include "pinocchio/algorithm/geometry.hpp"
#include "pinocchio/spatial/explog.hpp"
#include "pinocchio/spatial/se3.hpp"

namespace snakelib_estimation
{

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;

/**
 * Result of robust_central_direction().
 */
struct RobustDirectionResult
{
    std::optional<Vector3> mean;
    double agreement = 0.0;
    std::vector<std::size_t> rejected;
};

/**
 * Calculate the mean direction vector, rejecting outliers using the median.
 *
 * Equivalent to the Python:
 *     robust_central_direction()
 *
 * @param vectors          Input Nx3 vectors.
 * @param outlier_threshold Dot-product threshold for rejecting directions.
 *
 * @return mean       Normalized mean direction, or std::nullopt if degenerate.
 * @return agreement  Magnitude of the mean of surviving vectors.
 * @return rejected   Indices of rejected input vectors.
 */
inline RobustDirectionResult robustCentralDirection(
    const std::vector<Vector3>& vectors,
    double outlier_threshold)
{
    constexpr double min_norm = 1e-3;
    constexpr double min_mean_norm = 1e-9;

    RobustDirectionResult result;

    if (vectors.empty())
    {
        return result;
    }

    std::vector<std::size_t> accepted;
    accepted.reserve(vectors.size());

    // Reject vectors near zero.
    for (std::size_t i = 0; i < vectors.size(); ++i)
    {
        const double norm = vectors[i].norm();

        if (norm <= min_norm)
        {
            result.rejected.push_back(i);
        } else {
            accepted.push_back(i);
        }
    }

    // Python would produce an empty array here. There is no useful
    // direction to calculate, so return all vectors as rejected.
    if (accepted.empty())
    {
        result.rejected.clear();

        for (std::size_t i = 0; i < vectors.size(); ++i)
        {
            result.rejected.push_back(i);
        }

        return result;
    }

    // Normalize accepted vectors before computing the component-wise median.
    //
    // Note: the original Python code appears to intend unit vectors here,
    // although its implementation creates U directly from vectors.
    std::vector<Vector3> unit_vectors;
    unit_vectors.reserve(accepted.size());

    for (const std::size_t i : accepted)
    {
        unit_vectors.push_back(vectors[i].normalized());
    }

    // Component-wise median.
    Vector3 med;

    for (int component = 0; component < 3; ++component)
    {
        std::vector<double> values;
        values.reserve(unit_vectors.size());

        for (const auto& v : unit_vectors)
        {
            values.push_back(v[component]);
        }

        const auto middle = values.begin() + values.size() / 2;

        std::nth_element(
            values.begin(),
            middle,
            values.end());

        if (values.size() % 2 == 1)
        {
            med[component] = *middle;
        } else {
            const double upper = *middle;

            const auto lower_it =
                std::max_element(values.begin(), middle);

            med[component] = (*lower_it + upper) * 0.5;
        }
    }

    double med_norm = med.norm();

    // Degenerate median -> fall back to ordinary mean.
    if (med_norm < min_mean_norm)
    {
        med.setZero();

        for (const auto& v : unit_vectors)
        {
            med += v;
        }

        med /= static_cast<double>(unit_vectors.size());
        med_norm = med.norm();

        if (med_norm < min_mean_norm)
        {
            result.mean = std::nullopt;
            result.agreement = 0.0;
            result.rejected.clear();

            for (std::size_t i = 0; i < vectors.size(); ++i)
            {
                result.rejected.push_back(i);
            }

            return result;
        }
    }

    med /= med_norm;

    // Reject vectors whose direction differs too much from the median.
    std::vector<bool> keep(unit_vectors.size(), false);
    std::size_t num_kept = 0;

    for (std::size_t i = 0; i < unit_vectors.size(); ++i)
    {
        keep[i] =
            unit_vectors[i].dot(med) >= outlier_threshold;

        if (keep[i])
        {
            ++num_kept;
        }
    }

    // Too few survivors -> keep all.
    if (num_kept < 4)
    {
        std::fill(keep.begin(), keep.end(), true);
    } else {
        for (std::size_t i = 0; i < keep.size(); ++i)
        {
            if (!keep[i])
            {
                result.rejected.push_back(accepted[i]);
            }
        }
    }

    // Mean of surviving vectors.
    Vector3 mean_vec = Vector3::Zero();
    std::size_t survivor_count = 0;

    for (std::size_t i = 0; i < unit_vectors.size(); ++i)
    {
        if (keep[i])
        {
            mean_vec += unit_vectors[i];
            ++survivor_count;
        }
    }

    mean_vec /= static_cast<double>(survivor_count);

    const double agreement = mean_vec.norm();
    result.agreement = agreement;

    if (agreement < 1e-6)
    {
        result.mean = std::nullopt;
        return result;
    }

    result.mean = mean_vec / agreement;
    return result;
}


/**
 * Compute the geometric mean of a list of 3x3 rotation matrices.
 *
 * This uses iterative averaging in the tangent space of SO(3):
 */
inline Matrix3 meanOfRotations(
    const std::vector<Matrix3>& rotations,
    int max_iters = 10,
    double tol = 1e-6)
{
    if (rotations.empty())
    {
        throw std::invalid_argument("Can't take mean of empty rotation vector");
    }

    Matrix3 mean_R = rotations.front();

    for (int iteration = 0; iteration < max_iters; ++iteration)
    {
        Vector3 mean_error = Vector3::Zero();

        for (const auto& R : rotations)
        {
            // Pinocchio log3() returns the logarithm coordinates of SO(3).
            const Vector3 error = pinocchio::log3(mean_R.transpose() * R);

            mean_error += error;
        }

        mean_error /= static_cast<double>(rotations.size());

        // Converged.
        if (mean_error.norm() < tol)
        {
            break;
        }

        // Move toward the mean using the exponential map.
        mean_R = mean_R * pinocchio::exp3(mean_error);
    }

    return mean_R;
}


/**
 * Given an orientation and a desired X axis, calculate the rotation
 * around Z that aligns the frame with the desired X direction.
 */
inline Matrix3 getYawOffset(
    const pinocchio::SE3& orientation,
    const Vector3& desired_x)
{
    Vector3 x_ax = orientation.transpose() * desired_x;

    // Only the XY components are relevant.
    Eigen::Vector2d xy(x_ax.x(), x_ax.y());

    const double norm = xy.norm();

    if (norm < 1e-12)
    {
        return Matrix3::Identity();
    }

    xy /= norm;

    const double angle = std::atan2(-xy.y(), xy.x());
    const Vector3 axis = Vector3::UnitZ();

    Matrix3 yaw = Eigen::AngleAxisd(angle, axis).toRotationMatrix();

    return yaw.transpose();
}


/**
 * Estimate world pose from module feedback.
 *
 * Currently this is limited to orientation estimation.
 * Position must be specified separately.
 */
Estimator::Estimator()
        : world_(pinocchio::SE3::Identity())
{
}

/**
 * Update world orientation using module orientation measurements.
 *
 * @param orientations Rotation matrices expressed in the head frame.
 */
void Estimator::updateOrientations(
        const std::vector<Matrix3>& orientations)
{
    if (orientations.empty())
    {
        throw std::invalid_argument("'orientations' can't be empty");
    }

    // Initialize yaw offsets.
    if (yaw_offsets_.empty())
    {
        yaw_offsets_.resize(orientations.size(), std::nullopt);
    }

    // Handle a change in number of modules gracefully.
    if (yaw_offsets_.size() != orientations.size())
    {
        yaw_offsets_.resize(orientations.size(), std::nullopt);
    }

    std::vector<Matrix3> new_orientations;
    new_orientations.reserve(orientations.size());

    for (std::size_t i = 0; i < orientations.size(); ++i)
    {
        // Initialize yaw offset so that X points toward global X.
        if (!yaw_offsets_[i].has_value())
        {
	     pinocchio::SE3 orientation(orientations[i], Vector3::Zero());
            yaw_offsets_[i] = getYawOffset(orientation, Vector3::UnitX());
        }

        new_orientations.push_back(orientations[i] * yaw_offsets_[i].value());
    }

    // Estimate the common orientation.
    world_.rotation(meanOfRotations(new_orientations));

    // Update yaw offsets to compensate for modules drifting
    // relative to one another.
    for (std::size_t i = 0;
            i < orientations.size();
            ++i)
    {
        pinocchio::SE3 orientation(orientations[i], Vector3::Zero());

        yaw_offsets_[i] = getYawOffset(
            orientation
            world_.rotation().col(0)
        );
    }
}


/**
 * Update world orientation using IMU acceleration and angular velocity.
 *
 * @param lin_acc Linear acceleration vectors in the head frame.
 * @param ang_vel Angular velocity vectors in the head frame.
 * @param dt      Time interval over which velocity is integrated.
 */
void Estimator::updateImus(
        const std::vector<Vector3>& lin_acc,
        const std::vector<Vector3>& ang_vel,
        double dt)
{
    if (lin_acc.empty() || ang_vel.empty())
    {
        throw std::invalid_argument("'lin_acc' and 'ang_vel' can't be empty");
    } else if(lin_acc.size() != ang_vel.size()) {
        throw std::invalid_argument("'lin_acc' and 'ang_vel' are different lengths");
    }

    constexpr double outlier_deg = 10.0;

    const double outlier_threshold = std::cos(outlier_deg * M_PI / 180.0);

    // Estimate gravity direction.
    const Vector3 grav = robustCentralDirection(lin_acc, outlier_threshold);

    if (!gravity_result.mean.has_value())
    {
        return;
    }

    const Vector3 grav =
        gravity_result.mean.value();

    // Convert angular velocity * dt * 2 into roll/pitch/yaw
    // and construct the corresponding rotation.
    std::vector<Matrix3> angular_deltas;

    angular_deltas.reserve(ang_vel.size());

    for (const auto& angular_velocity : ang_vel)
    {
        // TODO: find out why i have to multiply this by 2
        const Vector3 rotation_vector = angular_velocity * dt * 2;
        angular_deltas.push_back(pinocchio::exp3(rotation_vector));
    }

    const Matrix3 ang_d = meanOfRotations(angular_deltas);

    const Matrix3 world_orientation =
        world_.rotation().transpose() *
        ang_d.transpose();

    // Rotate the measured Z axis toward gravity.
    const Vector3 current_z =
        world_orientation.col(2);

    Eigen::Quaterniond gravity_rotation;
    gravity_rotation.setFromTwoVectors(
        current_z,
        grav);

    world_.rotation(
        gravity_rotation.toRotationMatrix() *
        world_orientation);
}


/**
 * Set the world position manually.
 */
void Estimator::updatePosition(
        const Vector3& position)
{
    world_.translation(position);
}


/**
 * Access the estimated world transform.
 */
const pinocchio::SE3& Estimator::world() const
{
    return world_;
}


/**
 * Access to the estimated yaw offsets
 */
const std::vector<std::optional<Matrix3>>& Estimator::yaw_offsets() const
{
    return yaw_offsets_;
}

}  // namespace snakelib_estimation
