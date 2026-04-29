import expandTopicOrServiceName from "@/lib/ros/expandTopicName";
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

// Pre-defined commonly used mapping functions.
export const mappers = {
  in: inMappers,
  out: outMappers,
};

// Synced to rcl_interfaces/msg/ParameterType
export enum ParamType {
  BOOL = 1,
  INTEGER = 2,
  DOUBLE = 3,
  STRING = 4,
  BYTE_ARRAY = 5,
  BOOL_ARRAY = 6,
  INTEGER_ARRAY = 7,
  DOUBLE_ARRAY = 8,
  STRING_ARRAY = 9,
}

// Synced to rcl_interfaces/msg/ParameterValue
const typeFieldMap: Record<ParamType, string> = {
  [ParamType.BOOL]: "bool_value",
  [ParamType.INTEGER]: "integer_value",
  [ParamType.DOUBLE]: "double_value",
  [ParamType.STRING]: "string_value",
  [ParamType.BYTE_ARRAY]: "byte_array_value",
  [ParamType.BOOL_ARRAY]: "bool_array_value",
  [ParamType.INTEGER_ARRAY]: "integer_array_value",
  [ParamType.DOUBLE_ARRAY]: "double_array_value",
  [ParamType.STRING_ARRAY]: "string_array_value",
};

export default function useROSNode(nodeName: string) {
  const { connected: rosConnected, ros } = useROS();
  const [nodeConnected, setNodeConnected] = useState<boolean>(false);
  const connected = rosConnected && nodeConnected;

  useEffect(() => {
    setNodeConnected(ros.isNodeConnected(nodeName));

    const onNodeConnectedSub = ros.onNodeConnected(
      (connectedNodeName, nodeConnected) => {
        if (connectedNodeName === nodeName) {
          setNodeConnected(nodeConnected);
        }
      },
    );

    return () => onNodeConnectedSub.unsubscribe();
  }, [ros, nodeName, setNodeConnected]);

  function useSubscription<T>(
    path: string,
    idl: string,
    initial: T extends OutMapper ? ReturnType<T> : T,
    mapper?: T,
  ): T extends OutMapper ? ReturnType<T> : T {
    const [val, setVal] = useState(initial);

    useEffect(() => {
      // Note: roslibjs does not always resubscribe automatically after a WebSocket reconnection.
      // This affects latched topics because latched messages are only sent on new subscriptions,
      // not reconnections. Thus, we need to trigger a new subscription when the node reconnects.
      if (!connected) {
        return;
      }

      // Subscribe to the topic only when the node is connected, so that we create a new
      // subscription when the node reconnects.
      const topicName = expandTopicOrServiceName(path, nodeName);
      const callback = (msg: unknown) => {
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
      // We intentionally include [nodeName, ros, connected] to re-subscribe on reconnect.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [nodeName, ros, connected, path, idl, mapper]);

    return val;
  }

  function usePublisher<IN_T>(
    path: string,
    idl: string,
    mapper?: IN_T,
  ): (...a: IN_T extends InMapper ? Parameters<IN_T> : [IN_T]) => void {
    function _publish(...arg: any) {
      const topicName = expandTopicOrServiceName(path, nodeName);
      const data = typeof mapper === "function" ? mapper(...arg) : arg[0];

      // If the in mapper returns null/undefined, cancel the publish call.
      if (data == null) {
        return;
      }

      ros.publish(topicName, idl, data);
    }

    return useCallback(_publish, [nodeName, ros, path, idl, mapper]) as any;
  }

  function useService<IN_T, OUT_T>(
    path: string,
    idl: string,
    inMapper?: IN_T,
    outMapper?: OUT_T,
  ): (
    ...a: IN_T extends InMapper ? Parameters<IN_T> : [IN_T]
  ) => Promise<OUT_T extends OutMapper ? ReturnType<OUT_T> : OUT_T> {
    async function _service(...arg: any) {
      const serviceData =
        typeof inMapper === "function" ? inMapper(...arg) : arg[0];
      // If the in mapper returns null/undefined, cancel the service call.
      if (serviceData == null) {
        return;
      }

      const res = await ros.callService(
        expandTopicOrServiceName(path, nodeName),
        idl,
        serviceData,
      );
      return typeof outMapper === "function" ? outMapper(res) : res;
    }

    return useCallback(_service, [
      nodeName,
      ros,
      path,
      idl,
      inMapper,
      outMapper,
    ]) as any;
  }

  // Note: tried ROSLIB.Param but it does not work for array types. So instead, we
  // use a service call to the node's get_parameters service.
  function useGetParam<OUT_T>(
    paramName: string,
    outMapper?: OUT_T,
  ): () => Promise<OUT_T extends OutMapper ? ReturnType<OUT_T> : OUT_T> {
    async function _getParam() {
      const response = await ros.callService(
        expandTopicOrServiceName("~/get_parameters", nodeName),
        "rcl_interfaces/srv/GetParameters",
        { names: [paramName] },
      );

      const paramValue = (response as any).values[0];
      const typeField = typeFieldMap[paramValue.type as ParamType];
      const value = paramValue[typeField];

      return typeof outMapper === "function" ? outMapper(value) : value;
    }

    return useCallback(_getParam, [nodeName, ros, paramName, outMapper]) as any;
  }

  function useSetParam<IN_T>(
    paramName: string,
    paramType: ParamType,
    inMapper?: IN_T,
  ): (
    ...a: IN_T extends InMapper ? Parameters<IN_T> : [IN_T]
  ) => Promise<boolean> {
    async function _setParam(...arg: any) {
      const mappedValue =
        typeof inMapper === "function" ? inMapper(...arg) : arg[0];
      if (mappedValue == null) {
        return;
      }

      const typeField = typeFieldMap[paramType];
      const response = await ros.callService(
        expandTopicOrServiceName("~/set_parameters", nodeName),
        "rcl_interfaces/srv/SetParameters",
        {
          parameters: [
            {
              name: paramName,
              value: { type: paramType, [typeField]: mappedValue },
            },
          ],
        },
      );

      return (response as any).results[0].successful;
    }

    return useCallback(_setParam, [
      nodeName,
      ros,
      paramName,
      paramType,
      inMapper,
    ]) as any;
  }

  return useMemo(
    () => ({
      name: nodeName,
      connected,
      ros,
      useService,
      useSubscription,
      usePublisher,
      useSetParam,
      useGetParam,
    }),
    // Local hook functions close over nodeName/ros/connected, so omitting them from
    // deps is intentional
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [nodeName, ros, rosConnected, nodeConnected],
  );
}
