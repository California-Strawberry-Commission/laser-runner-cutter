import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";

export default function useROS() {
  const ros = useContext(ROSContext);
  const [connected, setConnected] = useState<boolean>(false);

  useEffect(() => {
    ros.onStateChange(() => {
      setConnected(ros.isConnected());
    });
    setConnected(ros.isConnected());
  }, [ros, setConnected]);

  return { connected, ros };
}
