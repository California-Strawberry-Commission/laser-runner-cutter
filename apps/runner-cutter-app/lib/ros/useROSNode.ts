import type { NodeInfo } from "@/lib/NodeInfo";
import useROS from "@/lib/ros/useROS";
import { useCallback, useEffect, useMemo, useState } from "react";
import expandTopicOrServiceName from "./expandTopicName";

// https://stackoverflow.com/questions/50011616/typescript-change-function-type-so-that-it-returns-new-value
type ReplaceReturnType<T extends (...a: any) => any, TNewReturn> = (...a: Parameters<T>) => TNewReturn;


export default function useROSNode<STATE_T>(nodeName: string, initalState: STATE_T) {
    const { nodeInfo: rosbridgeNodeInfo, ros } = useROS();
    const [nodeConnected, setNodeConnected] = useState<boolean>(false);
    const [nodeState, setNodeState] = useState(initalState);

    // Defines a service on this node.
    function service<
        IN_MAPPER_T extends (...a: any) => any,
        OUT_MAPPER_T extends (res: any) => any,
        T = ReturnType<OUT_MAPPER_T>,
    >(
        path: string,
        idl: string,
        in_mapper: IN_MAPPER_T,
        out_mapper: OUT_MAPPER_T
    ): (...a: Parameters<IN_MAPPER_T>) => T {

        const topic = expandTopicOrServiceName(path, nodeName);

        async function _service(...arg: any) {
            const service_data = in_mapper(...arg);
            const res = await node.ros.callService(topic, idl, service_data);
            return out_mapper(res)
        }

        return _service as any;
    }

    const node = useMemo(() => {
        return {
            name: nodeName,
            connected: rosbridgeNodeInfo.connected && nodeConnected,
            state: nodeState,
            ros,
            service
        };
    }, [nodeName, rosbridgeNodeInfo, nodeConnected, nodeState]);

    useEffect(() => {
        const onNodeConnectedSub = ros.onNodeConnected(
            (connectedNodeName, connected) => {
                if (connectedNodeName === nodeName) {
                    setNodeConnected(connected);
                }
            }
        );

        const stateSub = ros.subscribe(
            `${nodeName}/state`,
            "guidance_brain_interfaces/State",
            (message) => {
                setNodeState(message);
            }
        );

        return () => {
            onNodeConnectedSub.unsubscribe();
            stateSub.unsubscribe();
        };
    }, [ros, nodeName, setNodeConnected, setNodeState]);

    return node;
}