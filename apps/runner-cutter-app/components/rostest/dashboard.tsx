"use client";

import { useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";

export default function Dashboard() {
  const ros = useContext(ROSContext);
  const [rosConnected, setRosConnected] = useState<boolean>(
    ros.ros.isConnected
  );

  useEffect(() => {
    ros.onStateChange(() => {
      setRosConnected(ros.ros.isConnected);
    });
  }, [setRosConnected]);

  return (
    <div>
      <p className="text-center">{`rosConnected: ${rosConnected}`}</p>
    </div>
  );
}
