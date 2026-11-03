#include <optional>

#include "Eigen/Dense"
#include "pinocchio/algorithm/center-of-mass.hpp"
#include "pinocchio/algorithm/frames.hpp"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/spatial/se3.hpp"

#include "snakelib_estimation/kinematics.hpp"

namespace snakelib_estimation
{
    
// Static helper: Parses raw string data with a FreeFlyer root joint
pinocchio::Model Kinematics::parseUrdf(const std::string& urdf_xml)
{
    pinocchio::Model model;
    // Instantiate a FreeFlyer joint model
    pinocchio::JointModelFreeFlyer root_joint;
    
    // Pass the root joint type to the parser signature
    pinocchio::urdf::buildModelFromXML(urdf_xml, root_joint, model);
    
    return model;
}

// Public constructor
Kinematics::Kinematics(std::string urdf_xml)
    : Kinematics(parseUrdf(urdf_xml))
{}

// Private constructor
Kinematics::Kinematics(const pinocchio::Model& built_model)
    : model_(built_model),
      data_(model_),
      world_(pinocchio::SE3::Identity()),
      world_twist_(pinocchio::Motion::Zero()),
      n_joints_(model_.nv - 2)
{}

void Kinematics::update(
        Eigen::VectorXd position,
        std::optional<Eigen::Ref<const Eigen::VectorXd>> velocity)
{
    // Allocate the full composite configuration vector q
    Eigen::VectorXd q = Eigen::VectorXd::Zero(model_.nq);
    
    // Pack spatial transform of this->_world: [translation, rotation]
    q.head<3>() = world_.translation();
    Eigen::Quaterniond quat(world_.rotation());
    q.segment<4>(3) = quat.coeffs(); 

    // Append our joint angles
    q.tail(position.size()) = position;

    if (velocity.has_value()) {
        Eigen::VectorXd v = Eigen::VectorXd::Zero(model_.nv);
        // Convert world_twist_ from world frame to the local moving base frame
        // pinocchio::Motion automatically handles the spatial transform mapping
        pinocchio::Motion local_twist = world_.actInv(world_twist_);
        // Pack spatial velocity: [linear_velocity, angular_velocity]
        v.head<3>() = local_twist.linear();
        v.tail<3>() = local_twist.angular();

        // Append the actuated joint velocities right after the 6-DoF base (index 6)
        v.tail(velocity->size()) = *velocity;

        // Run position and velocity forward kinematics
        pinocchio::forwardKinematics(model_, data_, q, v);
    } else
    {
        pinocchio::forwardKinematics(model_, data_, q);
    }
    pinocchio::updateFramePlacements(model_, data_);
}

Vector3 Kinematics::com()
{
    return pinocchio::centerOfMass(model_, data_);
}

std::vector<Matrix4> Kinematics::fk(
    const std::vector<std::string>& joint_names,
    const std::string& suffix)
{
    std::vector<std::string> names = joint_names;
    if (names.empty()) {
        if (model_.names.size() <= 2)
        {
            throw std::runtime_error("Model uninitialized or has zero joints");
        }
        // Skip index 0 ("universe") AND index 1 (the FreeFlyer root)
        names.assign(model_.names.begin() + 2, model_.names.end());
    }
    std::vector<Matrix4> retval;
    retval.reserve(names.size());
    for (int i = 0; i < names.size(); ++i)
    {
        retval[i] = data_.oMf[model_.getFrameId(names[i] + suffix)]
                         .toHomogeneousMatrix();
    }
    return retval;
}

std::vector<Vector3> Kinematics::imu_to_head_frame(
    const std::vector<Vector3>& imu_vals)
{
    // Assert the size matches the number of joints
    if (imu_vals.size() != static_cast<size_t>(n_joints_)) {
        throw std::invalid_argument(
            "ValueError: The number of IMU values (" + std::to_string(imu_vals.size()) + 
            ") does not match the number of joints (" + std::to_string(n_joints_) + ")."
        );
    }

    // Retrieve the forward kinematics transformations
    std::vector<Matrix4> transforms = fk({}, "_imu");
    
    // Allocate space for the output vector
    std::vector<Vector3> rotated_vals;
    rotated_vals.reserve(n_joints_);

    // Apply each joint's 3D rotation matrix to the corresponding input vector
    for (int i = 0; i < n_joints_; ++i) {
        rotated_vals.push_back(transforms[i].block<3, 3>(0, 0) * imu_vals[i]);
    }

    return rotated_vals;
}

}  // namespace snakelib_estimation