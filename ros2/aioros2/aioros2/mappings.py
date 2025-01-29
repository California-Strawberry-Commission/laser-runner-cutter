from typing import List
import rclpy

# Maps python types to a ROS parameter integer enum
dataclass_ros_map = {
    bool: 1,
    int: 2,
    float: 3,
    str: 4,
    bytes: 5,
    List[bool]: 6,
    List[int]: 7,
    List[float]: 8,
    List[str]: 9,
}

# Maps python types to a ROS parameter enum
dataclass_ros_enum_map = {
    bool: rclpy.Parameter.Type.BOOL,
    int: rclpy.Parameter.Type.INTEGER,
    float: rclpy.Parameter.Type.DOUBLE,
    str: rclpy.Parameter.Type.STRING,
    bytes: rclpy.Parameter.Type.BYTE_ARRAY,
    List[bool]: rclpy.Parameter.Type.BOOL_ARRAY,
    List[int]: rclpy.Parameter.Type.INTEGER_ARRAY,
    List[float]: rclpy.Parameter.Type.DOUBLE_ARRAY,
    List[str]: rclpy.Parameter.Type.STRING_ARRAY,
}

# Maps ROS types to a corresponding attribute containing the
# typed value
ros_type_getter_map = {
    1: "bool_value",
    2: "integer_value",
    3: "double_value",
    4: "string_value",
    5: "byte_array_value",
    6: "bool_array_value",
    7: "integer_array_value",
    8: "double_array_value",
    9: "string_array_value",
}
