import { createContext } from "react";
import ROS from "@/lib/ros/ROS";

const ROSContext = createContext(new ROS("ws://localhost:9090"));
export default ROSContext;
