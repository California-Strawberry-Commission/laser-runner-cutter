import useROSNode, { mappers } from "@/lib/ros/useROSNode";

export default function useGuidanceBrainNode(nodeName: string) {
  const node = useROSNode(nodeName);

  const state = node.useSubscription(
    "~/state",
    "furrow_perceiver_interfaces/State",
    {
      guidance_offset: 0,
    }
  );

  const setGuidanceOffset = node.useService(
    "~/set_guidance_offset",
    "common_interfaces/SetInt32",
    mappers.in.number, // maps request message to a JS api & solidifies typing info
    mappers.out.success // maps incoming response & solidifies typing info
  );

  return {
    ...node,
    state,
    setGuidanceOffset,
  };
}
