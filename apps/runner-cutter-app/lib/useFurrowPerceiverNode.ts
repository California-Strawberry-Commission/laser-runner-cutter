import { useCallback, useEffect, useMemo, useState } from "react";
import useROSNode from "@/lib/ros/useROSNode";

const INITIAL_STATE = {
  guidance_offset: 0
};

export default function useGuidanceBrainNode(nodeName: string) {
  const node = useROSNode(nodeName);

  const state = node.useTopic("~/state", "furrow_perceiver_interfaces/State", INITIAL_STATE);

  const setGuidanceOffset = node.useService(
    "~/set_guidance_offset",
    "common_interfaces/SetInt32",
    (offset: number) => ({ data: offset }), // maps request message to a JS api & solidifies typing info
    (_data: any) => undefined, // maps incoming response & solidifies typing info
  );


  return {
    ...node,
    state,
    setGuidanceOffset
  };
}
