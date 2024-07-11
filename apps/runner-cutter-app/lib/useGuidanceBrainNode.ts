import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";
import useROSNode from "./ros/useROSNode";

const INITIAL_STATE = {
  guidance_active: false,
  amiga_connected: false,
  follower_pid: { p: 0, i: 0, d: 0 }
};

export default function useGuidanceBrainNode(nodeName: string) {
  const node = useROSNode(nodeName, INITIAL_STATE);

  const setActive = node.service(
    "~/set_active",
    "std_srvs/SetBool",
    (active: boolean) => ({ data: active }), // maps request message to a JS api & solidifies typing info
    (_data) => null, // maps incoming response & solidifies typing info
  );

  const setPID = node.service(
    "~/set_follower_pid",
    "guidance_brain_interfaces/SetPID",
    (p: number, i: number, d: number) => ({ pid: { p, i, d } }),
    () => undefined,
  )

  return {
    ...node,
    setActive,
    setPID
  };
}
