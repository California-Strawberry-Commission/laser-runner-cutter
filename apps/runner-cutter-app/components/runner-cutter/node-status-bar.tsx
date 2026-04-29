"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { NodeInfo } from "@/lib/NodeInfo";
import { cn } from "@/lib/utils";
import { ChevronDown } from "lucide-react";
import { useState } from "react";

export default function NodeStatusBar({
  nodeInfos,
  className,
}: {
  nodeInfos: NodeInfo[];
  className?: string;
}) {
  const [selectedNode, setSelectedNode] = useState<NodeInfo | null>(null);

  const connectedCount = nodeInfos.filter((n) => n.connected).length;
  const allConnected = connectedCount === nodeInfos.length;

  return (
    <>
      <Popover>
        <PopoverTrigger asChild>
          <Button
            className={cn(
              "flex items-center gap-2 px-3 py-1.5",
              allConnected
                ? "bg-green-500 text-white hover:bg-green-600"
                : "bg-red-500 text-white hover:bg-red-600",
              className,
            )}
          >
            <span
              className={cn(
                "h-2 w-2 rounded-full flex-none bg-white",
                !allConnected && "animate-pulse",
              )}
            />
            {allConnected
              ? `All ${nodeInfos.length} nodes connected`
              : `${connectedCount} / ${nodeInfos.length} nodes connected`}
            <ChevronDown className="h-3 w-3 ml-1" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-64 p-2" align="start">
          <div className="flex flex-col gap-1">
            {nodeInfos.map((node) => (
              <button
                key={node.name}
                className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted text-sm text-left w-full"
                onClick={() => setSelectedNode(node)}
              >
                <span
                  className={cn(
                    "h-2 w-2 rounded-full flex-none",
                    node.connected ? "bg-green-500" : "bg-red-500",
                  )}
                />
                <span className="truncate">{node.name}</span>
                <span className="ml-auto text-xs text-muted-foreground shrink-0">
                  {node.connected ? "Connected" : "Disconnected"}
                </span>
              </button>
            ))}
          </div>
        </PopoverContent>
      </Popover>
      <Dialog
        open={selectedNode !== null}
        onOpenChange={(open) => !open && setSelectedNode(null)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>{selectedNode?.name}</DialogTitle>
            <DialogDescription>
              {selectedNode?.connected ? "Connected" : "Disconnected"}
            </DialogDescription>
          </DialogHeader>
          <pre className="overflow-auto text-xs max-h-96">
            {JSON.stringify(selectedNode?.state ?? {}, undefined, 2)}
          </pre>
        </DialogContent>
      </Dialog>
    </>
  );
}
