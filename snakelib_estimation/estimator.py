from collections.abc import Iterable

import numpy as np
import pinocchio as pin


def robust_central_direction(vectors: Iterable[Iterable[float]], outlier_threshold: float):
    """Calculate the mean direction vectors, rejecting outliers using the median.

    Original courtesy of Harry Shi

    Args:
        vectors (NxM): Iterable of N vectors to average.
        outlier_threshold (float): Joint velocities

    Returns:
        mean: Vector of size (M,) representing the average or None
        agreement: Estimate of how close non-rejected vectors were to each other
        rejected: Index of vectors not used to calculate the mean
    """
    # Reject vectors near zero
    min_norm = 1e-3
    norms = np.linalg.norm(vectors, axis=1)
    rejected = np.argwhere(norms <= min_norm).flatten().tolist()
    accepted = np.argwhere(norms > min_norm).flatten().tolist()
    U = np.array([np.divide(vectors[i], norms[i]) for i in range(len(vectors)) if i in accepted])

    # Robust central direction: component-wise median of the unit vectors,
    # renormalized. Dominated by the agreeing majority, so outliers don't move it.
    med = np.median(U, axis=0)
    mn = np.linalg.norm(med)
    if mn < 1e-9:  # degenerate median -> fall back
        med = U.mean(axis=0)
        mn = np.linalg.norm(med)
        if mn < 1e-9:
            return None, 0.0, list(range(len(vectors)))
    med = med / mn

    keep = (U @ med) >= outlier_threshold
    if int(keep.sum()) < 4:  # too few survivors -> keep all
        keep = np.ones(len(U), dtype=bool)
    rejected += [accepted[j] for j in range(len(accepted)) if not keep[j]]

    mean_vec = U[keep].mean(axis=0)
    agreement = float(np.linalg.norm(mean_vec))  # over survivors
    if agreement < 1e-6:
        return None, agreement, rejected

    return mean_vec / agreement, agreement, rejected


def mean_of_rotations(rotations, max_iters=10, tol=1e-6):
    """Compute the geometric mean of a list of 3x3 rotation matrices."""
    # Start with the first rotation as our initial guess
    mean_R = rotations[0]

    for _ in range(max_iters):
        # Compute the velocity vectors (tangent space error vectors) from mean to each rotation
        tangent_errors = []
        for R in rotations:
            # pin.log3 computes the 3D angular error vector (Lie algebra)
            error_vector = pin.log3(mean_R.T @ R)
            tangent_errors.append(error_vector)

        # Average the vectors in the flat tangent space
        mean_error = np.mean(tangent_errors, axis=0)

        # If the update is negligible, we've converged
        if np.linalg.norm(mean_error) < tol:
            break

        # Step towards the mean using the exponential map
        mean_R = mean_R @ pin.exp3(mean_error)

    return mean_R


def get_yaw_offset(orientation: pin.SE3, desired_x):
    """Given an orientation and a desired X axis, get the rotation along Z that
    will rotate the frame to match"""
    x_ax = (orientation.T @ desired_x)[0:2]
    x_ax /= np.linalg.norm(x_ax)
    ang = np.arctan2(np.linalg.det([x_ax, [1,0]]), np.dot(x_ax, [1,0]))
    return pin.rpy.rpyToMatrix(0,0,ang).T

class Estimator:
    """Estimate world pose from module feedback.

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
    """

    def __init__(self):
        # Offsets for 'orientation' feedback
        self._yaw_offsets = None
        # Estimated world transform
        self.world = pin.SE3.Identity()

    def update_orientations(self, orientations):
        """Update self.world using orientation data from the modules.

        Args:
            orientations: 3x3 Rotation matrices specified in the head frame
        """
        if self._yaw_offsets is None:
            self._yaw_offsets = [None] * len(orientations)

        new_orientations = []

        for i, mat in enumerate(orientations):
            # Initialize yaw offsets to point to global x
            if self._yaw_offsets[i] is None:
                self._yaw_offsets[i] = get_yaw_offset(mat, [1, 0, 0])

            new_orientations.append(mat @ self._yaw_offsets[i])

        self.world.rotation = mean_of_rotations(new_orientations)
        # Update yaw offsets to correct for modules drifting in relation to eachother
        for i, mat in enumerate(orientations):
            self._yaw_offsets[i] = get_yaw_offset(mat, self.world.rotation[:,0])

    def update_imus(self, lin_acc, ang_vel, dt):
        """Update self.world using IMU data from the modules.

        Args:
            lin_acc: Linear acceleration specified in the head frame
            ang_vel: Angular velocities specified in the head frame
            dt: time across which this velocity should be applied
        """
        # Get gravity direction
        outlier_deg = 10
        grav, _, _ = robust_central_direction(lin_acc,  np.cos(np.radians(outlier_deg)))

        ang_ds = [pin.exp3(a * dt * 2) for a in ang_vel]
        ang_d = mean_of_rotations(ang_ds)
        world_orientation = self.world.inverse().rotation @ ang_d.T
        grav_d = pin.Quaternion()
        pin.Quaternion.setFromTwoVectors(grav_d, world_orientation[0:3, 2], grav)
        self.world.rotation = grav_d.matrix() @ world_orientation

    def update_position(self, position):
        """Set the world position manually."""
        self.world.translation = position
