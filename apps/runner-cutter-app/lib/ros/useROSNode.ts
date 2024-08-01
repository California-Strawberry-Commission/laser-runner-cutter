import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";
import expandTopicOrServiceName from "./expandTopicName";

type in_mapper_t = (...a: any) => any;
type out_mapper_t = (res: any) => any;

/**
 * Creates a topic client using a mapper and inferred types.
 * @param nodeName 
 * @param ros 
 * @returns 
 */
function useTopic(nodeName: string, ros: any) {
    return function <T>(
        path: string,
        idl: string,
        initial: T extends out_mapper_t ? ReturnType<T> : T,
        mapper?: T
    ): T extends out_mapper_t ? ReturnType<T> : T {
        const topic = expandTopicOrServiceName(path, nodeName);
        const [val, setVal] = useState(initial);

        useEffect(() => {
            const sub = ros.subscribe(topic, idl, (v: T) => setVal(typeof mapper == "function" ? mapper(v) : v));
            return () => sub.unsubscribe();

        }, [nodeName, ros, path, idl]);

        return val;
    }
}


/**
 * Creates a service API with mappers, using inferred types.
 * @param nodeName 
 * @param ros 
 * @returns 
 */

function useService(nodeName: string, ros: any) {
    type mappable_fn_t<IN_T, OUT_T> = (...a: IN_T extends in_mapper_t ? Parameters<IN_T> : [IN_T]) => Promise<OUT_T extends out_mapper_t ? ReturnType<OUT_T> : OUT_T>
    
    // Main type signature - accepts "mapper" fns to make TS api cleaner
    function _service<
        IN_T,
        OUT_T,
    >(
        path: string,
        idl: string,
        in_mapper?: IN_T,
        out_mapper?: OUT_T
    ): mappable_fn_t<IN_T, OUT_T> {
        const topic = expandTopicOrServiceName(path, nodeName);

        async function _service(...arg: any) {
            const service_data = typeof in_mapper == 'function' ? in_mapper(...arg) : arg[0];
            const res = await ros.callService(topic, idl, service_data);
            return typeof out_mapper == 'function' ? out_mapper(res) : res;
        }

        return useCallback(_service, [path, idl, nodeName, ros]) as any;
    }

    return _service
}

/**
 * Creates a service API without mappers, using explicit types.
 * @param nodeName 
 * @param ros 
 * @returns 
 */
function useRawService(nodeName: string, ros: any) {
    function _service<D, R>(
        path: string,
        idl: string,
    ): (val: D) => Promise<R> {
        const topic = expandTopicOrServiceName(path, nodeName);

        async function _service(arg: D) {
            return ros.callService(topic, idl, arg);
        }

        return useCallback(_service, [path, idl, nodeName, ros]) as any;
    }

    return _service
}

export default function useROSNode<STATE_T>(nodeName: string) {
    const { nodeInfo: rosbridgeNodeInfo, ros } = useROS();
    const [nodeConnected, setNodeConnected] = useState<boolean>(false);

    useEffect(() => {
        const onNodeConnectedSub = ros.onNodeConnected(
            (connectedNodeName, connected) => {
                if (connectedNodeName === nodeName) {
                    setNodeConnected(connected);
                }
            }
        );

        return () => onNodeConnectedSub.unsubscribe();
    }, [ros, nodeName, setNodeConnected]);

    const api = {
        useRawService: useRawService(nodeName, ros),
        useService: useService(nodeName, ros),
        useTopic: useTopic(nodeName, ros),
    }

    return useMemo(() => {
        return {
            name: nodeName,
            connected: rosbridgeNodeInfo.connected && nodeConnected,
            ros,
            ...api
        };
    }, [nodeName, rosbridgeNodeInfo, nodeConnected, api]);
}