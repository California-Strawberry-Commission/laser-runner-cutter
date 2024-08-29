import useROSNode, {mappers} from "@/lib/ros/useROSNode";

const INITIAL_STATE = {
  guidance_offset: 0
};

export default function useGuidanceBrainNode(nodeName: string) {
  const node = useROSNode(nodeName);

  const state = node.useTopic("~/state", "furrow_perceiver_interfaces/State", INITIAL_STATE);

  const setGuidanceOffset = node.useService(
    "~/set_guidance_offset",
    "common_interfaces/SetInt32",
    mappers.in.number, // maps request message to a JS api & solidifies typing info
    mappers.out.success, // maps incoming response & solidifies typing info
  );


  return {
    ...node,
    state,
    setGuidanceOffset
  };
}
