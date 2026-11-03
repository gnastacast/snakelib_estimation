#pragma once

#include <vector>
#include <optional>

#include "Eigen/Dense"
#include "pinocchio/algorithm/model.hpp"
#include "pinocchio/spatial/se3.hpp"

namespace snakelib_estimation
{

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using Matrix4 = Eigen::Matrix4d;

class Kinematics
{
public:
    Kinematics(std::string urdf_xml);

    void update(
        Eigen::VectorXd position,
        std::optional<Eigen::Ref<const Eigen::VectorXd>>  velocity);

    Vector3 com();

    std::vector<Matrix4> fk(
        const std::vector<std::string>& joint_names = {},
        const std::string& suffix = "_link");

    std::vector<Vector3> imu_to_head_frame(
        const std::vector<Vector3>& imu_vals
    );

private:
    // Private delegating constructor used internally
    Kinematics(const pinocchio::Model& built_model);

    // Static helper to handle Pinocchio's in-place parsing signature
    static pinocchio::Model parseUrdf(const std::string& urdf_xml);

    pinocchio::Model model_;
    pinocchio::Data data_;
    pinocchio::SE3 world_;
    pinocchio::Motion world_twist_;
    int n_joints_;
};

}  // namespace snakelib_estimation
