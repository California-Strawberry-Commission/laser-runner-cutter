import useROS from "@/lib/ros/useROS";
import { useEffect } from "react";
import { toast } from "sonner";

const LogLevel = {
  DEBUG: 10,
  INFO: 20,
  WARN: 30,
  ERROR: 40,
  FATAL: 50,
};

export default function useNotifications(topic: string) {
  const { ros } = useROS();

  useEffect(() => {
    const sub = ros.subscribe(topic, "rcl_interfaces/Log", (message) => {
      const msg = message["msg"];
      switch (message["level"]) {
        case LogLevel.WARN:
          toast.warning(msg);
          break;
        case LogLevel.ERROR:
        case LogLevel.FATAL:
          toast.error(msg);
          break;
        default:
          toast.info(msg);
          break;
      }
    });
    return () => sub.unsubscribe();
  }, [ros, topic]);
}
