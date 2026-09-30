#pragma once

#include <vector>
#include <optional>

#include "Eigen/Dense"
#include "pinocchio/spatial/se3.hpp"

namespace snakelib_estimation
{

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;

class Estimator
{
public:
    Estimator();

    void updateImus(
        const std::vector<Vector3>& lin_acc,
        const std::vector<Vector3>& ang_vel,
        double dt);

    void updateOrientations(
        const std::vector<Matrix3>& orientations);

    void updatePosition(
        const Vector3& position);

    const pinocchio::SE3& world() const;

    pinocchio::SE3& world();

private:
    std::vector<std::optional<Matrix3>> yaw_offsets_;
    pinocchio::SE3 world_;
};

}  // namespace snakelib_estimation
