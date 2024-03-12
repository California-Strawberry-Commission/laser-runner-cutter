import { createContext } from "react";
import ROSLIB from "roslib";

const ROSContext = createContext(
  new ROSLIB.Ros({ url: "ws://localhost:9090" })
);
export default ROSContext;
