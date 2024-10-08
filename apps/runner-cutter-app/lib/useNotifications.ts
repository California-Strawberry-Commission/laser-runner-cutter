import useROS from "@/lib/ros/useROS";
import { useEffect } from "react";
import { toast } from "sonner";

export default function useNotifications(topic: string) {
  const { ros } = useROS();

  useEffect(() => {
    const sub = ros.subscribe(topic, "std_msgs/String", (message) => {
      toast(message["data"]);
    });
    return () => sub.unsubscribe();
  }, [ros, topic]);
}
