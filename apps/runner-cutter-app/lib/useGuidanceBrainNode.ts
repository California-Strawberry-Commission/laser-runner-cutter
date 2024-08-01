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

type in_mapper_t = (...a: any) => any;
type out_mapper_t = (res: any) => any;
function us(nodeName: string, ros: any) {

  // Main type signature - accepts "mapper" fns to make TS api cleaner
  function _service<
    IN_T,
    OUT_T,
  >(
    path: string,
    idl: string,

  ): (...a: IN_T extends in_mapper_t ? Parameters<IN_T> : [IN_T]) => Promise<OUT_T extends out_mapper_t ? ReturnType<OUT_T> : OUT_T> {

    async function _service(...arg: any) {

    }

    return useCallback(_service, [path, idl, nodeName, ros]) as any;
  }

  return _service
}


export default function useGuidanceBrainNode(nodeName: string) {
  const node = useROSNode(nodeName);

  const state = node.useTopic("~/state", "guidance_brain_interfaces/State", INITIAL_STATE);
  
  /* Example of unmapped service
  const doService = node.useService<{data: boolean}, {success: boolean}>(
    "~/set_active",
    "std_srvs/SetBool"
  )
  */

  const setActive = node.useService(
    "~/set_active",
    "std_srvs/SetBool",
    (active: boolean) => ({ data: active }), // maps request message to a JS api & solidifies typing info
    (_data: any) => undefined, // maps incoming response & solidifies typing info
  );

  const setP = node.useService(
    "~/set_p",
    "common_interfaces/SetFloat32",
    (p: number) => ({ data: p }),
    () => undefined,
  );

  const setI = node.useService(
    "~/set_i",
    "common_interfaces/SetFloat32",
    (i: number) => ({ data: i }),
    () => undefined,
  );

  const setD = node.useService(
    "~/set_d",
    "common_interfaces/SetFloat32",
    (d: number) => ({ data: d }),
    () => undefined,
  );

  const setSpeed = node.useService(
    "~/set_speed",
    "common_interfaces/SetFloat32",
    (speed: number) => ({ data: speed }),
    () => undefined,
  );

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
