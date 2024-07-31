import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";
import useROSNode from "@/lib/ros/useROSNode";

const INITIAL_STATE = {
  guidance_active: false,
  amiga_connected: false,

  speed: 0,
  follower_pid: { p: 0, i: 0, d: 0 },

  perceiver_valid: false,
  error: 0,
  command: 0,
};

export default function useGuidanceBrainNode(nodeName: string) {
  const node = useROSNode(nodeName);

  const state = node.useTopic("~/state", "guidance_brain_interfaces/State", INITIAL_STATE);
  
  const setActive = node.useService(
    "~/set_active",
    "std_srvs/SetBool",
    (active: boolean) => ({ data: active }), // maps request message to a JS api & solidifies typing info
    (_data) => null, // maps incoming response & solidifies typing info
  );

  const setP = node.useService(
    "~/set_p",
    "common_interfaces/SetFloat32",
    (p: number) => ({ data: p }),
    () => undefined,
  )

  const setI = node.useService(
    "~/set_i",
    "common_interfaces/SetFloat32",
    (i: number) => ({ data: i }),
    () => undefined,
  )

  const setD = node.useService(
    "~/set_d",
    "common_interfaces/SetFloat32",
    (d: number) => ({ data: d }),
    () => undefined,
  )

  const setSpeed = node.useService(
    "~/set_speed",
    "common_interfaces/SetFloat32",
    (speed: number) => ({ data: speed }),
    () => undefined,
  )

  return {
    ...node,
    state,
    setActive,
    setP,
    setI,
    setD,
    setSpeed,
  };
}
