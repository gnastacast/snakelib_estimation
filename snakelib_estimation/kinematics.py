from collections.abc import Iterable

import numpy as np
import pinocchio as pin


class Kinematics:
    """Kinematics for snake robots.

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
    """

    def __init__(self, urdf: str):
        """Kinematics are built from a URDF loaded from a file or robot_description."""
        self._model = pin.buildModelFromXML(urdf, pin.JointModelFreeFlyer())
        self._joint_names = self._model.names[2:]
        self._n_joints = len(self._joint_names)
        self._data = self._model.createData()
        self._world = pin.SE3.Identity()
        self._world_twist = pin.Motion.Zero()

    def update(self, position: Iterable[float], velocity: Iterable[float] = None):
        """Update the kinematics using module feedback.

        Args:
            position (Iterable[float]): Joint angles.
            velocity (Iterable[float]): Joint velocities

        Raises:
            AssertionError: If the position or velocity are the wrong size
        """
        q = np.array(position, dtype=np.float64)
        assert q.shape == (self._n_joints,), f'Position shape {q.shape} not {(self._n_joints,)}'
        q = np.hstack([
            self._world.translation,
            pin.Quaternion(self._world.rotation).coeffs(),
            q
        ])
        if velocity is not None:
            v = np.array(velocity, dtype=np.float64)
            v = np.hstack([self._world_twist.linear, self._world_twist.angular, v])
            assert len(v) == len(q) - 1, f'Velocity length {len(v)} not equal to {len(q)} - 1'
            pin.forwardKinematics(self._model, self._data, q, v)
        else:
            pin.forwardKinematics(self._model, self._data, q)
        pin.updateFramePlacements(self._model, self._data)

    def com(self):
        """Calculate the center of mass of the robot."""
        return pin.centerOfMass(self._model, self._data)

    def fk(self, names: Iterable[str] | None = None, suffix='_link'):
        """Calculate the forward kinematics of each module in the world frame.

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
        """
        retval = []
        if names is None:
            names = self._joint_names
        for name in names:
            frame_id = self._model.getFrameId(f'{name}{suffix}')
            retval.append(self._data.oMf[frame_id])
        return retval

    def imu_to_head_frame(self, imu_vals: Iterable[float]):
        """Rotates imu_vals from IMU frame into head frame.

        This only works after calling self.update.
        This does not apply the translation component of each transformation.

        Args:
            imu_vals: Values such as lin_accel and ang_vel specified in the IMU frame

        Returns:
            A np array of the values rotated into the head frame

        Raises:
            AssertionError: When the number of imu_values does not match the joints.
        """
        assert len(imu_vals) == self._n_joints
        rotations = [f.rotation for f in self.fk(suffix='_imu')]
        return np.einsum('nij,nj->ni', rotations, imu_vals)
