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
 * Compute the mean direction of a collection of vectors.
 *
 * This uses an iterative weighted average to reject outliers
 */
Vector3 robustCentralDirection(
    const std::vector<Vector3>& vectors,
    double outlier_threshold,
    double tol = 1e-9)
{
    double min_norm = 1e-6;


    std::vector<Vector3> accepted;
    accepted.reserve(vectors.size());

    Vector3 mean = Vector3::Zero();

    for (const auto& v : vectors) {
        double norm = v.norm();
        if (norm < min_norm) {
            continue;
        }
        accepted.push_back(v.normalized());
        mean += accepted.back();
    }

    int N  = accepted.size();
    if (N == 0) {
        return Vector3::Zero();
    }

    // Check if vectors completely cancel each other out initially
    if (mean.isZero(min_norm)) {
        return Eigen::Vector3d::Zero();
    }

    mean.normalize();
    Eigen::Vector3d prev_mean = mean;
    std::vector<double> weights(N, 1.0);

    for (int iter = 0; iter < 10; ++iter) {
        mean.setZero();

        for (size_t i = 0; i < N; ++i) {
            // Directional similarity
            double cos_theta = accepted[i].dot(prev_mean);

            if (cos_theta < outlier_threshold) {
                // Dampen weight linearly as deviation grows.
                // We add a tiny epsilon to avoid division by zero if vectors are orthogonal.
                double angle_dist = 1.0 - cos_theta;
                double threshold_dist = 1.0 - outlier_threshold;
                weights[i] = threshold_dist / std::max(angle_dist, 1e-5);
            } else {
                // Inlier
                weights[i] = 1.0;
            }

            mean += weights[i] * vectors[i];
        }

        if (mean.isZero(min_norm)) {
            break; // Edge case: Weights caused a perfect cancelation
        }
        mean.normalize();

        // Check for convergence
        double change = 1.0 - mean.dot(prev_mean);
        if (change < tol) {
            break;
        }
        prev_mean = mean;
    }
    return mean;
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
    const Matrix3& orientation,
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
            yaw_offsets_[i] = getYawOffset(orientations[i], Vector3::UnitX());
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
        yaw_offsets_[i] = getYawOffset(
            orientations[i],
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

    if (grav.isZero(1e-9))
    {
        return;
    }

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
