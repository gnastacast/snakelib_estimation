# Snake estimation package

This package aims to build tools to better estimate the state of the snake.
It mainly consists of the `Kinematics` and `Estimation` classes. It also includes
a node to publish state estimates as the `world` tf which can be run using

`ros2 run snakelib_estimation state_estimation_node`
