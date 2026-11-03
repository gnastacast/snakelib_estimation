#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "snakelib_estimation/estimator.hpp"
#include "snakelib_estimation/kinematics.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_snakelib_estimation, m)
{
    m.doc() = "Python bindings for snake pose estimation";

    // Kinematics

    py::class_<snakelib_estimation::Kinematics>(m, "Kinematics", R"pbdoc(
        Kinematics for snake robots.

        Typical usage example (msg is a HebiSensors message):

            kin = Kinematics()
            kin.update(msg.position, msg.velocity)
            frames = kin.fk()
            translations = np.array([f.translation for f in frames]
            rotations = np.array([f.rotation for f in frames]
            accelerations = kin.imu_to_head_frame(zip(
                msg.lin_acc.x,
                msg.lin_acc.y,
                msg.lin_acc.z
            ))
    )pbdoc")

        .def(py::init<std::string>(), py::arg("urdf_xml"),
             "Constructs the Kinematics model from a URDF XML string.")

        .def("update", &snakelib_estimation::Kinematics::update,
             py::arg("position"),
             py::arg("velocity") = std::nullopt,
             R"pbdoc(
                Update the kinematics using module feedback.

                Args:
                    position (Iterable[float]): Joint angles.
                    velocity (Iterable[float]): Joint velocities

                Raises:
                    RuntimeError: If the position or velocity are the wrong size
             )pbdoc")

        .def("com", &snakelib_estimation::Kinematics::com,
             "Calculate the center of mass of the robot.")

        .def("fk", &snakelib_estimation::Kinematics::fk,
             py::arg("joint_names") = std::vector<std::string>{},
             py::arg("suffix") = "_link",
             R"pbdoc(
                Calculate the forward kinematics of each module in the world frame.

                This only works after calling self.update
                The world frame is the same as head_link unless self._world is updated

                Args:
                    names: Joint names. Will default to all joints.
                    suffix: A suffix to attach to each joint name that represents a link
                    in the URDF.

                Returns:
                    A list of pin.SE3 representing the transformation of each joint

                Raises:
                    KeyError: Any name + suffix not in the URDF.
        )pbdoc")

        .def("imu_to_head_frame", &snakelib_estimation::Kinematics::imu_to_head_frame,
             py::arg("imu_vals"),
             R"pbdoc(
                Rotates imu_vals from IMU frame into head frame.

                This only works after calling self.update.
                This does not apply the translation component of each transformation.

                Args:
                    imu_vals: Values such as lin_accel and ang_vel specified in the IMU frame

                Returns:
                    A np array of the values rotated into the head frame

                Raises:
                    AssertionError: When the number of imu_values does not match the joints.
             )pbdoc");

    // Estimator

    py::class_<snakelib_estimation::Estimator>(m, "Estimator", R"pbdoc(
        Estimate world pose from module feedback.

        For now, this is limited to orientation estimation.
        Position must be specified separately.

        Typical usage example (m is a HebiSensors message):
            est = Estimator()
            kin = snakelib_estimation.Kinematics()
            kin.update(m.position, m.velocity)
            #### Method 1
            lin_acc = kin.imu_to_head_frame(list(zip(m.lin_acc.x, m.lin_acc.y, m.lin_acc.z)))
            ang_vel = kin.imu_to_head_frame(list(zip(m.ang_vel.x, m.ang_vel.y, m.ang_vel.z)))
            est.update_imus(lin_acc, ang_vel, 0.01)
            #### Method 2
            ori = msg.orientation
            rots = [pin.Quaternion(*q).matrix() for q in zip(ori.w, ori.x, ori.y, ori.z)]
            rots = [f.rotation @ o for o, f in zip(rots, kin.fk())]
            est.update_orientations(rots)
    )pbdoc")

        .def(
            py::init<>(),
            "Create a new pose estimator."
        )

        .def(
            "update_imus",
            &snakelib_estimation::Estimator::updateImus,
            py::arg("lin_acc"),
            py::arg("ang_vel"),
            py::arg("dt"),
            R"pbdoc(
                Update the world orientation using IMU measurements.

                Parameters
                ----------
                lin_acc : array-like, shape (N, 3)
                    Linear acceleration measurements in the head frames.

                ang_vel : array-like, shape (N, 3)
                    Angular velocity measurements in rad/s.

                dt : float
                    Time interval in seconds.

                The angular velocity is integrated using:

                    R = pinocchio.exp3(angular_velocity * dt)
            )pbdoc"
        )

        .def(
            "update_orientations",
            &snakelib_estimation::Estimator::updateOrientations,
            py::arg("orientations"),
            R"pbdoc(
                Update the world orientation using module orientations.

                Parameters
                ----------
                orientations : array-like, shape (N, 3, 3)
                    Rotation matrices expressed in the head frames.
            )pbdoc"
        )

        .def(
            "update_position",
            &snakelib_estimation::Estimator::updatePosition,
            py::arg("position"),
            R"pbdoc(
                Set the world position.

                Parameters
                ----------
                position : array-like, shape (3,)
                    World position.
            )pbdoc"
        )

        .def(
            "world",
            [](const snakelib_estimation::Estimator& self)
            {
                return self.world().toHomogeneousMatrix();
            },
            R"pbdoc(
                Return the estimated world transform.

                Returns
                -------
                numpy.ndarray
                    4x4 homogeneous transformation matrix.
            )pbdoc"
        )

        .def(
            "yaw_offsets",
            [](const snakelib_estimation::Estimator& self)
            {
                return self.yaw_offsets();
            },
            R"pbdoc(
                Return the estimated yaw offsets.

                Returns
                -------
                array-like
                    3x3 rotation matrices.
            )pbdoc"
        );
}
