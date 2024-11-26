from aioros2 import serve_nodes
from laser_control.laser_control_node import LaserControlNode
from camera_control.camera_control_node import CameraControlNode
from runner_cutter_control.runner_cutter_control_node import RunnerCutterControlNode
from rclpy.parameter import Parameter


def main():
    # aioros2 currently links node imports using node parameters (see
    # aioros2.launch_driver.LaunchNode._link_node()). Because we are not using
    # aioros2's LaunchNode to launch these nodes, we manually override the
    # parameters here.
    serve_nodes(
        LaserControlNode(name="laser0"),
        CameraControlNode(name="camera0"),
        RunnerCutterControlNode(
            name="control0",
            parameter_overrides=[
                Parameter(
                    "laser_node.name",
                    Parameter.Type.STRING,
                    "laser0",
                ),
                Parameter(
                    "camera_node.name",
                    Parameter.Type.STRING,
                    "camera0",
                ),
            ],
        ),
    )


if __name__ == "__main__":
    main()
