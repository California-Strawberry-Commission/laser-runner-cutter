import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

const LASER_STATES = ["disconnected", "connecting", "stopped", "playing"];
const INITIAL_STATE = LASER_STATES[0];

export default function useLaserNode(nodeName: string) {
  const { nodeInfo: rosbridgeNodeInfo, ros } = useROS();
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);
  const [laserState, setLaserState] = useState<string>(LASER_STATES[0]);

  const nodeInfo: NodeInfo = useMemo(() => {
    return {
      name: nodeName,
      connected: rosbridgeNodeInfo.connected && nodeConnected,
      state: { laserState },
    };
  }, [nodeName, rosbridgeNodeInfo, nodeConnected, laserState]);

  const getState = useCallback(async () => {
    const result = await ros.callService(
      `${nodeName}/get_state`,
      "laser_control_interfaces/GetState",
      {}
    );
    setLaserState(LASER_STATES[result.state.data]);
  }, [ros, nodeName, setLaserState]);

  // Initial node state
  useEffect(() => {
    const connected = ros.isNodeConnected(nodeName);
    if (connected) {
      getState();
    }
    setNodeConnected(connected);
  }, [ros, nodeName, getState, setNodeConnected]);

  // Subscriptions
  useEffect(() => {
    const onNodeConnectedSub = ros.onNodeConnected(
      (connectedNodeName, connected) => {
        if (connectedNodeName === nodeName) {
          setNodeConnected(connected);
          if (connected) {
            getState();
          } else {
            setLaserState(INITIAL_STATE);
          }
        }
      }
    );

    const stateSub = ros.subscribe(
      `${nodeName}/state`,
      "laser_control_interfaces/State",
      (message) => {
        setLaserState(LASER_STATES[message.data]);
      }
    );

    return () => {
      onNodeConnectedSub.unsubscribe();
      stateSub.unsubscribe();
    };
  }, [ros, nodeName, getState, setNodeConnected, setLaserState]);

  const startDevice = useCallback(() => {
    // Optimistically set device state to "connecting"
    setLaserState(LASER_STATES[1]);
    ros.callService(`${nodeName}/start_device`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const closeDevice = useCallback(() => {
    ros.callService(`${nodeName}/close_device`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const addPoint = useCallback(
    (x: number, y: number) => {
      ros.callService(
        `${nodeName}/add_point`,
        "laser_control_interfaces/AddPoint",
        {
          point: {
            x: x,
            y: y,
          },
        }
      );
    },
    [ros, nodeName]
  );

  const clearPoints = useCallback(() => {
    ros.callService(`${nodeName}/clear_points`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const play = useCallback(() => {
    ros.callService(`${nodeName}/play`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const stop = useCallback(() => {
    ros.callService(`${nodeName}/stop`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const setColor = useCallback(
    (r: number, g: number, b: number) => {
      ros.callService(
        `${nodeName}/set_color`,
        "laser_control_interfaces/SetColor",
        {
          r: r,
          g: g,
          b: b,
          i: 0.0,
        }
      );
    },
    [ros, nodeName]
  );

  return {
    nodeInfo,
    laserState,
    startDevice,
    closeDevice,
    addPoint,
    clearPoints,
    play,
    stop,
    setColor,
  };
}
