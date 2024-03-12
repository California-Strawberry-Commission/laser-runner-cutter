import useROS from "@/lib/ros/useROS";

export default function useControlNode(nodeName: string) {
  const { callService, subscribe } = useROS();

  const calibrate = () => {
    callService(`${nodeName}/calibrate`, "std_srvs/Empty", {});
  };

  return { calibrate };
}
