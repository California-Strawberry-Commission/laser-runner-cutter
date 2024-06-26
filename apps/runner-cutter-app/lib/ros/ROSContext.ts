import { createContext } from "react";
import ROS from "@/lib/ros/ROS";

const ROSContext = createContext(
  new ROS(process.env.NEXT_PUBLIC_ROSBRIDGE_URL ?? "ws://localhost:9090")
);
export default ROSContext;
