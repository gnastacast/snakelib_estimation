#!/usr/bin/env python3
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
import numpy as np
import pinocchio as pin
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from snakelib_estimation import Estimator, Kinematics
from snakelib_msgs.msg import HebiSensors
from std_msgs.msg import ColorRGBA, Header, String
from tf2_ros import TransformBroadcaster, TransformStamped
from visualization_msgs.msg import Marker


def pin_to_tf(frame: pin.SE3, frame_id: str, header: Header) -> TransformStamped:
    q = pin.Quaternion(frame.rotation)

    t_stamped = TransformStamped()
    t_stamped.header = header
    t_stamped.child_frame_id = frame_id

    # Set translation
    t_stamped.transform.translation.x = float(frame.translation[0])
    t_stamped.transform.translation.y = float(frame.translation[1])
    t_stamped.transform.translation.z = float(frame.translation[2])

    # Set rotation
    t_stamped.transform.rotation.x = float(q.x)
    t_stamped.transform.rotation.y = float(q.y)
    t_stamped.transform.rotation.z = float(q.z)
    t_stamped.transform.rotation.w = float(q.w)

    return t_stamped


class EstimatorNode(Node):
    def __init__(self):
        super().__init__('snake_estimator')

        # Get debug parameter
        self.declare_parameter('draw_debug', False)

        self.last_t = None
        self.kinematics = None
        self.estimator = Estimator()
        self.sensor_sub = self.create_subscription(
            HebiSensors,
            '/hebi_sensors/data',
            self.hebi_cb,
            1
        )

        # Critical: Define a QoS profile that matches the publisher
        qos_profile = QoSProfile(
            depth=1,
            history=HistoryPolicy.KEEP_LAST,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL  # <-- Essential line
        )

        self.robot_description_sub = self.create_subscription(
            String,
            '/robot_description',
            self.robot_description_cb,
            qos_profile
        )

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        if self.get_parameter('draw_debug').get_parameter_value().bool_value:
            self.debug_pub = self.create_publisher(
                Marker,
                '/state_estimate_debug',
                100,
            )

            self.debug_msg = Marker()
            self.debug_msg.lifetime = Duration(sec=0, nanosec=500000000)
            self.debug_msg.header.frame_id = 'head'
            self.debug_msg.type = Marker.LINE_LIST
            self.debug_msg.id = 0
            self.debug_msg.action = Marker.ADD
            self.debug_msg.color.r = 1.0
            self.debug_msg.color.g = 1.0
            self.debug_msg.color.b = 1.0
            self.debug_msg.color.a = 1.0
            self.debug_msg.scale.x = 0.01
            self.debug_msg.scale.y = 0.01
            self.debug_msg.scale.z = 0.01

        else:
            self.debug_pub = None
            self.debug_msg = None

    def draw_accelerations(self, accelerations, gravity=None, scale=0.01):
        positions = [f.translation for f in self.kinematics.fk(suffix='_imu')]
        for i, (accel, p) in enumerate(zip(accelerations, positions)):
            self.debug_msg.points.append(Point(x=p[0], y=p[1], z=p[2]))
            self.debug_msg.points.append(Point(
                x=float(accel[0] * scale + p[0]),
                y=float(accel[1] * scale + p[1]),
                z=float(accel[2] * scale + p[2])
            ))
            self.debug_msg.colors.append(ColorRGBA(r=1.0, a=1.0))
            self.debug_msg.colors.append(ColorRGBA(r=1.0, a=1.0))

        if gravity is not None:
            self.debug_msg.points.append(Point(x=0.0, y=0.0, z=0.0))
            self.debug_msg.points.append(Point(
                x=float(gravity[0] * scale * 2),
                y=float(gravity[1] * scale * 2),
                z=float(gravity[2] * scale * 2)
            ))
            self.debug_msg.colors.append(ColorRGBA(g=1.0, a=1.0))
            self.debug_msg.colors.append(ColorRGBA(g=1.0, a=1.0))

    def draw_frames(self, orientations, scale=0.1):
        positions = [f.translation for f in self.kinematics.fk(suffix='_imu')]
        xs = [o[:, 0] * scale for o in orientations[0:]]
        ys = [o[:, 1] * scale for o in orientations[0:]]
        zs = [o[:, 2] * scale for o in orientations[0:]]

        for i, (x, p) in enumerate(zip(xs, positions)):
            self.debug_msg.points.append(Point(x=p[0], y=p[1], z=p[2]))
            self.debug_msg.points.append(Point(
                x=float(x[0] + p[0]),
                y=float(x[1] + p[1]),
                z=float(x[2] + p[2])
            ))
            self.debug_msg.colors.append(ColorRGBA(r=1.0, a=1.0))
            self.debug_msg.colors.append(ColorRGBA(r=1.0, a=1.0))
        for i, (y, p) in enumerate(zip(ys, positions)):
            self.debug_msg.points.append(Point(x=p[0], y=p[1], z=p[2]))
            self.debug_msg.points.append(Point(
                x=float(y[0] + p[0]),
                y=float(y[1] + p[1]),
                z=float(y[2] + p[2])
            ))
            self.debug_msg.colors.append(ColorRGBA(g=1.0, a=1.0))
            self.debug_msg.colors.append(ColorRGBA(g=1.0, a=1.0))
        for i, (z, p) in enumerate(zip(zs, positions)):
            self.debug_msg.points.append(Point(x=p[0], y=p[1], z=p[2]))
            self.debug_msg.points.append(Point(
                x=float(z[0] + p[0]),
                y=float(z[1] + p[1]),
                z=float(z[2] + p[2])
            ))
            self.debug_msg.colors.append(ColorRGBA(b=1.0, a=1.0))
            self.debug_msg.colors.append(ColorRGBA(b=1.0, a=1.0))

    def robot_description_cb(self, msg: String):
        self.kinematics = Kinematics(msg.data)

    def hebi_cb(self, msg: HebiSensors):
        if self.kinematics is None:
            return
        if self.debug_msg is not None:
            self.debug_msg.points.clear()
            self.debug_msg.colors.clear()
        self.kinematics.update(msg.position, msg.velocity)
        if np.any(np.isnan(msg.orientation.x)):
            # Not receiving orientations. Have to build our own
            t = rclpy.time.Time.from_msg(msg.header.stamp)
            if self.last_t is None:
                self.last_t = t
            dt = (t - self.last_t).nanoseconds / 1e9
            if dt > 0.5:
                dt = 0
            lin_acc = self.kinematics.imu_to_head_frame(list(zip(
                msg.lin_acc.x,
                msg.lin_acc.y,
                msg.lin_acc.z
            )))
            ang_vel = self.kinematics.imu_to_head_frame(list(zip(
                msg.ang_vel.x,
                msg.ang_vel.y,
                msg.ang_vel.z
            )))
            self.estimator.update_imus(lin_acc, ang_vel, 0.01)
            if self.debug_msg is not None:
                self.draw_accelerations(lin_acc)

        else:
            msg.orientation = msg.orientation
            rots = [pin.Quaternion(*q).matrix() for q in zip(
                msg.orientation.w,
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z
            )]
            for i in range(len(rots)):
                # TODO: Figure out why this is needed
                rots[i] = rots[i] @ pin.rpy.rpyToMatrix(0, np.pi/2, 0)
                rots[i] = pin.rpy.rpyToMatrix(np.pi/2, 0, np.pi) @ rots[i]
            # Apply kinematics
            rots = [f.rotation @ o for o, f in zip(rots, self.kinematics.fk())]
            self.estimator.update_orientations(rots)
            if self.debug_msg is not None:
                self.draw_frames([r @ o for r, o in zip(rots, self.estimator.yaw_offsets())])

        if self.debug_msg is not None:
            self.debug_msg.header = msg.header
            self.debug_msg.header.frame_id = 'head_link'
            self.debug_pub.publish(self.debug_msg)

        # Set estimator position to kinematic CoM
        self.estimator.update_position(self.kinematics.com())

        header_world = msg.header
        header_world.frame_id = 'world'
        self.tf_broadcaster.sendTransform(pin_to_tf(
            pin.SE3(self.estimator.world()).inverse(),
            # self.estimator.world.inverse(),
            '/head_link',
            header_world
        ))


def main(args=None):
    rclpy.init(args=args)

    snake_estimator = EstimatorNode()

    rclpy.spin(snake_estimator)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    snake_estimator.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
