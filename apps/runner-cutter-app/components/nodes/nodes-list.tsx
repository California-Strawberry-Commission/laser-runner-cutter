"use client";

import React, { useCallback, useContext, useEffect, useState } from "react";
import ROSContext from "@/lib/ros/ROSContext";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";

type Node = {
  name: string;
  connected: boolean;
};

const NODES = ["/control0", "/camera0", "/laser0"];

export default function NodesList() {
  const ros = useContext(ROSContext);
  const [nodes, setNodes] = useState<Node[]>([]);

  const updateNodes = useCallback(() => {
    const connectedNodes = ros.getNodes();
    const nodeStatuses = NODES.map((nodeName) => {
      return {
        name: nodeName,
        connected: ros.isConnected() && connectedNodes.includes(nodeName),
      };
    });
    nodeStatuses.unshift({ name: "Rosbridge", connected: ros.isConnected() });
    setNodes(nodeStatuses);
  }, [ros, setNodes]);

  // Initial node statuses
  useEffect(() => {
    updateNodes();
  }, [updateNodes]);

  // Subscriptions
  useEffect(() => {
    ros.onStateChange((state) => {
      updateNodes();
    });
    ros.onNodeConnected((nodeName, connected) => {
      if (NODES.includes(nodeName)) {
        updateNodes();
      }
    });
    // TODO: unsubscribe from ros.onNodeConnected and ros.onStateChange
  }, [ros, updateNodes]);

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Name</TableHead>
          <TableHead className="w-36">Status</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {nodes.map((node) => (
          <TableRow
            key={node.name}
            onClick={() => {
              // TODO: show details modal if node is connected
            }}
          >
            <TableCell>{node.name}</TableCell>
            <TableCell
              className={`${
                node.connected ? "text-green-600" : "text-red-600"
              }`}
            >
              {node.connected ? "Connected" : "Disconnected"}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}
