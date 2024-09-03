import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";
import expandTopicOrServiceName from "@/lib/ros/expandTopicName";

type in_mapper_t = (...a: any) => any;
type out_mapper_t = (res: any) => any;

/**
 * Creates a topic subscriptions with optional mapper, using either inferred or explicit types.
 * @param nodeName
 * @param ros
 * @returns
 */
function useTopic(nodeName: string, ros: any) {
  return function useForLint<T>(
    path: string,
    idl: string,
    initial: T extends out_mapper_t ? ReturnType<T> : T,
    mapper?: T
  ): T extends out_mapper_t ? ReturnType<T> : T {
    const [val, setVal] = useState(initial);

    useEffect(() => {
      const topic = expandTopicOrServiceName(path, nodeName);

      const sub = ros.subscribe(topic, idl, (v: T) =>
        setVal(typeof mapper === "function" ? mapper(v) : v)
      );
      return () => sub.unsubscribe();
    }, [nodeName, ros, path, idl, mapper]);

    return val;
  };
}

/**
 * Creates a service API with mappers, using either inferred or explicit types.
 * @param nodeName
 * @param ros
 * @returns
 */
function useService(nodeName: string, ros: any) {
  type mappable_fn_t<IN_T, OUT_T> = (
    ...a: IN_T extends in_mapper_t ? Parameters<IN_T> : [IN_T]
  ) => Promise<OUT_T extends out_mapper_t ? ReturnType<OUT_T> : OUT_T>;

  // Main type signature - accepts "mapper" fns to make TS api cleaner
  // Name function
  return function useForLint<IN_T, OUT_T>(
    path: string,
    idl: string,
    in_mapper?: IN_T,
    out_mapper?: OUT_T
  ): mappable_fn_t<IN_T, OUT_T> {
    async function _service(...arg: any) {
      const serviceName = expandTopicOrServiceName(path, nodeName);

      const service_data =
        typeof in_mapper === "function" ? in_mapper(...arg) : arg[0];
      const res = await ros.callService(serviceName, idl, service_data);
      return typeof out_mapper === "function" ? out_mapper(res) : res;
    }

    return useCallback(_service, [path, idl, in_mapper, out_mapper]) as any;
  };
}

export default function useROSNode(nodeName: string) {
  const { connected: rosConnected, ros } = useROS();
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);

  useEffect(() => {
    setNodeConnected(ros.isNodeConnected(nodeName));

    const onNodeConnectedSub = ros.onNodeConnected(
      (connectedNodeName, connected) => {
        if (connectedNodeName === nodeName) {
          setNodeConnected(connected);
        }
      }
    );

    return () => onNodeConnectedSub.unsubscribe();
  }, [ros, nodeName, setNodeConnected]);

  const us = useService(nodeName, ros);
  const ut = useTopic(nodeName, ros);

  return useMemo(() => {
    return {
      name: nodeName,
      connected: rosConnected && nodeConnected,
      ros,
      useService: us,
      useTopic: ut,
    };
  }, [ros, nodeName, rosConnected, nodeConnected, us, ut]);
}
