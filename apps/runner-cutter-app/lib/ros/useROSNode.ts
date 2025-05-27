import expandTopicOrServiceName from "@/lib/ros/expandTopicName";
import ROS from "@/lib/ros/ROS";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";

type InMapper = (...a: any) => any;
type OutMapper = (res: any) => any;

const inMappers = {
  noArg: () => ({}),
  trigger: () => ({}),
  number: (d: number) => (d == null || isNaN(d) ? null : { data: d }), // Nulls will NOT result in a set
};

const outMappers = {
  noArg: (_data: any) => undefined,
  success: (_data: any) => _data.success as boolean,
  successWithMessage: (_data: any) =>
    _data as { success: boolean; message: string },
  trigger: (_data: any) => _data as { success: boolean; message: string },
};

/**
 * Pre-defined commonly used mapping functions.
 */
export const mappers = {
  in: inMappers,
  out: outMappers,
};

/**
 * Creates a topic subscription with optional mapper, using either inferred or explicit types.
 * @param nodeName
 * @param ros
 * @param nodeConnected
 * @returns
 */
function useSubscription(nodeName: string, ros: ROS, nodeConnected: boolean) {
  return function useForLint<T>(
    path: string,
    idl: string,
    initial: T extends OutMapper ? ReturnType<T> : T,
    mapper?: T
  ): T extends OutMapper ? ReturnType<T> : T {
    const [val, setVal] = useState(initial);

    useEffect(() => {
      // Note: roslibjs does not always resubscribe automatically after a WebSocket reconnection.
      // This affects latched topics because latched messages are only sent on new subscriptions,
      // not reconections. Thus, we need to trigger a new subscription when the node reconnects.
      if (!nodeConnected) {
        return;
      }

      // Subscribe to the topic only when the node is connected, so that we create a new
      // subscription when the node reconnects
      const topicName = expandTopicOrServiceName(path, nodeName);

      const callback = (msg: ROSLIB.Message) => {
        const val = typeof mapper === "function" ? mapper(msg) : msg;
        setVal(val);
      };
      const sub = ros.subscribe(topicName, idl, callback);

      return () => {
        // roslibjs does not clear the listeners when calling unsubscribe() without the callback.
        // Thus, make sure to unsubscribe the specific callback so that we do not end up with
        // multiple registrations of the same callback.
        sub.unsubscribe(callback);
      };
    }, [nodeName, ros, path, idl, mapper, nodeConnected]);

    return val;
  };
}

/**
 * Creates a topic publisher with optional input mapper, using either inferred or explicit types.
 * @param nodeName
 * @param ros
 * @returns
 */
function usePublisher(nodeName: string, ros: ROS) {
  type MappableFn<IN_T> = (
    ...a: IN_T extends InMapper ? Parameters<IN_T> : [IN_T]
  ) => void;

  return function useForLint<IN_T>(
    path: string,
    idl: string,
    mapper?: IN_T
  ): MappableFn<IN_T> {
    function _publish(...arg: any) {
      const topicName = expandTopicOrServiceName(path, nodeName);
      const data = typeof mapper === "function" ? mapper(...arg) : arg[0];

      // If the in mapper returns null/undefined, cancel the publish call.
      if (data == null) {
        return;
      }

      ros.publish(topicName, idl, data);
    }

    return useCallback(_publish, [path, idl, mapper]) as any;
  };
}

/**
 * Creates a service API with mappers, using either inferred or explicit types.
 * @param nodeName
 * @param ros
 * @returns
 */
function useService(nodeName: string, ros: ROS) {
  type MappableFn<IN_T, OUT_T> = (
    ...a: IN_T extends InMapper ? Parameters<IN_T> : [IN_T]
  ) => Promise<OUT_T extends OutMapper ? ReturnType<OUT_T> : OUT_T>;

  // Main type signature - accepts "mapper" fns to make TS api cleaner
  // Name function
  return function useForLint<IN_T, OUT_T>(
    path: string,
    idl: string,
    inMapper?: IN_T,
    outMapper?: OUT_T
  ): MappableFn<IN_T, OUT_T> {
    async function _service(...arg: any) {
      const serviceName = expandTopicOrServiceName(path, nodeName);

      const serviceData =
        typeof inMapper === "function" ? inMapper(...arg) : arg[0];

      // If the in mapper returns null/undefined, cancel the service call.
      if (serviceData == null) {
        return;
      }

      const res = await ros.callService(serviceName, idl, serviceData);
      return typeof outMapper === "function" ? outMapper(res) : res;
    }

    return useCallback(_service, [path, idl, inMapper, outMapper]) as any;
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

  const useServiceFn = useService(nodeName, ros);
  const useSubscriptionFn = useSubscription(
    nodeName,
    ros,
    rosConnected && nodeConnected
  );
  const usePublisherFn = usePublisher(nodeName, ros);

  return useMemo(() => {
    return {
      name: nodeName,
      connected: rosConnected && nodeConnected,
      ros,
      useService: useServiceFn,
      useSubscription: useSubscriptionFn,
      usePublisher: usePublisherFn,
    };
  }, [
    ros,
    nodeName,
    rosConnected,
    nodeConnected,
    useServiceFn,
    useSubscriptionFn,
    usePublisherFn,
  ]);
}
