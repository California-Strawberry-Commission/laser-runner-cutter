import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";
import expandTopicOrServiceName from "./expandTopicName";

// https://stackoverflow.com/questions/50011616/typescript-change-function-type-so-that-it-returns-new-value
type ReplaceReturnType<T extends (...a: any) => any, TNewReturn> = (...a: Parameters<T>) => TNewReturn;


type stateOptions<STATE_T> = {
    initalState: STATE_T,
    stateIdl: string,
    stateTopic?: string,
}

export default function useROSNode<STATE_T>(nodeName: string) {
    const { nodeInfo: rosbridgeNodeInfo, ros } = useROS();
    const [nodeConnected, setNodeConnected] = useState<boolean>(false);

    // Put ros and name on `this` to pass to subsequent `topic` and `service` calls.
    // Ugly hack but what do you expect from react ¯\_(ツ)_/¯
    const ctx = {
        ros: ros,
        nodeName: nodeName
    };


    // Defines a service on this node.
    function _service<
        IN_MAPPER_T extends (...a: any) => any,
        OUT_MAPPER_T extends (res: any) => any,
        T = ReturnType<OUT_MAPPER_T>,
    >(
        this: typeof ctx,
        path: string,
        idl: string,
        in_mapper: IN_MAPPER_T,
        out_mapper: OUT_MAPPER_T
    ): (...a: Parameters<IN_MAPPER_T>) => T {
        const topic = expandTopicOrServiceName(path, this.nodeName);

        async function _service(...arg: any) {
            const service_data = in_mapper(...arg);
            const res = await ros.callService(topic, idl, service_data);
            return out_mapper(res)
        }

        return useCallback(_service, [path, idl, in_mapper, out_mapper, this.nodeName, this.ros]) as any;
    }

    const useService = _service.bind(ctx);

    // Defines a service on this node.
    function _topic<T>(
        this: typeof ctx,
        path: string,
        idl: string,
        initial: T
    ): T {
        const topic = expandTopicOrServiceName(path, this.nodeName);
        const [val, setVal] = useState(initial);

        useEffect(() => {
            const sub = this.ros.subscribe(topic, idl, (v) => {
                setVal(v);
            });

            return () => sub.unsubscribe();

        }, [this.ros, this.nodeName, path, idl]);

        return val;
    }

    const useTopic = _topic.bind(ctx);

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


    return useMemo(() => {
        return {
            name: nodeName,
            connected: rosbridgeNodeInfo.connected && nodeConnected,
            ros,
            useService,
            useTopic
        };
    }, [nodeName, rosbridgeNodeInfo, nodeConnected]);
}