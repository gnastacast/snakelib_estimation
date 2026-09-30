#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "snakelib_estimation/estimator.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_snakelib_estimation, m)
{
    m.doc() = "Python bindings for the C++ pose estimator";

    // -------------------------------------------------------------------------
    // Estimator
    // -------------------------------------------------------------------------

    py::class_<snakelib_estimation::Estimator>(m, "Estimator")
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
            "rotation",
            [](const snakelib_estimation::Estimator& self)
            {
                return self.world().rotation();
            },
            R"pbdoc(
                Return the estimated world rotation.

                Returns
                -------
                numpy.ndarray
                    3x3 rotation matrix.
            )pbdoc"
        )

        .def(
            "position",
            [](const snakelib_estimation::Estimator& self)
            {
                return self.world().translation();
            },
            R"pbdoc(
                Return the estimated world position.

                Returns
                -------
                numpy.ndarray
                    3-element position vector.
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
        );
}
