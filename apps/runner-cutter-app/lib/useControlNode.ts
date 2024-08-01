import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

export type State = {
  calibrated: boolean;
  state: string;
  tracks: Track[];
};
export type Track = {
  id: number;
  normalizedPixelCoords: { x: number; y: number };
  state: TrackState;
};
export enum TrackState {
  Pending,
  Active,
  Completed,
  Failed,
}

function convertStateMessage(message: any): State {
  return {
    calibrated: message.calibrated,
    state: message.state,
    tracks: message.tracks.map(convertTrackMessage),
  };
}

function convertTrackMessage(message: any): Track {
  return {
    id: message.id,
    normalizedPixelCoords: message.normalized_pixel_coords,
    state: message.state as TrackState,
  };
}

export default function useControlNode(nodeName: string) {
  const { nodeInfo: rosbridgeNodeInfo, ros } = useROS();
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);
  const [nodeState, setNodeState] = useState<State>({
    calibrated: false,
    state: "idle",
    tracks: [],
  });

  const nodeInfo: NodeInfo = useMemo(() => {
    return {
      name: nodeName,
      connected: rosbridgeNodeInfo.connected && nodeConnected,
      state: nodeState,
    };
  }, [nodeName, rosbridgeNodeInfo, nodeConnected, nodeState]);

  // Subscriptions
  useEffect(() => {
    const onNodeConnectedSub = ros.onNodeConnected(
      (connectedNodeName, connected) => {
        if (connectedNodeName === nodeName) {
          setNodeConnected(connected);
        }
      }
    );

    const stateSub = ros.subscribe(
      `${nodeName}/state`,
      "runner_cutter_control_interfaces/State",
      (message) => {
        setNodeState(convertStateMessage(message));
      }
    );

    return () => {
      onNodeConnectedSub.unsubscribe();
      stateSub.unsubscribe();
    };
  }, [ros, nodeName, setNodeConnected, setNodeState]);

  const calibrate = useCallback(() => {
    ros.callService(`${nodeName}/calibrate`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const addCalibrationPoint = useCallback(
    (normalizedX: number, normalizedY: number) => {
      ros.callService(
        `${nodeName}/add_calibration_points`,
        "runner_cutter_control_interfaces/AddCalibrationPoints",
        { normalized_pixel_coords: [{ x: normalizedX, y: normalizedY }] }
      );
    },
    [ros, nodeName]
  );

  const manualTargetAimLaser = useCallback(
    (normalizedX: number, normalizedY: number) => {
      ros.callService(
        `${nodeName}/manual_target_aim_laser`,
        "runner_cutter_control_interfaces/ManualTargetAimLaser",
        { normalized_pixel_coord: { x: normalizedX, y: normalizedY } }
      );
    },
    [ros, nodeName]
  );

  const startRunnerCutter = useCallback(() => {
    ros.callService(`${nodeName}/start_runner_cutter`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  const stop = useCallback(() => {
    ros.callService(`${nodeName}/stop`, "std_srvs/Trigger", {});
  }, [ros, nodeName]);

  return {
    nodeInfo,
    nodeState,
    calibrate,
    addCalibrationPoint,
    manualTargetAimLaser,
    startRunnerCutter,
    stop,
  };
}
