"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import type { NodeInfo } from "@/lib/NodeInfo";
import { cn } from "@/lib/utils";
import { ChevronDown, RotateCw } from "lucide-react";
import { useState } from "react";

export type RestartTarget = {
  label: string;
  onRestart: () => void;
};

export default function NodeStatusBar({
  nodeInfos,
  className,
  onRestartNodes,
  onRebootSystem,
  restartDisabled,
  restartTargets,
}: {
  nodeInfos: NodeInfo[];
  className?: string;
  onRestartNodes?: () => void;
  onRebootSystem?: () => void;
  restartDisabled?: boolean;
  restartTargets?: RestartTarget[];
}) {
  const [selectedNode, setSelectedNode] = useState<NodeInfo | null>(null);
  const [restartDialogOpen, setRestartDialogOpen] = useState(false);
  const [popoverOpen, setPopoverOpen] = useState(false);

  const connectedCount = nodeInfos.filter((n) => n.connected).length;
  const allConnected = connectedCount === nodeInfos.length;

  return (
    <>
      <Popover open={popoverOpen} onOpenChange={setPopoverOpen}>
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
            {restartTargets && restartTargets.length > 0 && (
              <>
                <Separator className="my-1" />
                <span className="px-2 py-1 text-xs text-muted-foreground">
                  Restart node
                </span>
                {restartTargets.map((target) => (
                  <button
                    key={target.label}
                    className="flex items-center gap-2 px-2 py-1.5 rounded hover:bg-muted text-sm text-left w-full disabled:opacity-50"
                    disabled={restartDisabled}
                    onClick={() => {
                      target.onRestart();
                      setPopoverOpen(false);
                    }}
                  >
                    <RotateCw className="h-3 w-3 flex-none" />
                    <span className="truncate">{target.label}</span>
                  </button>
                ))}
              </>
            )}
            {(onRestartNodes || onRebootSystem) && (
              <>
                <Separator className="mb-1" />
                <Button
                  className="w-full"
                  variant="destructive"
                  size="sm"
                  disabled={restartDisabled}
                  onClick={() => setRestartDialogOpen(true)}
                >
                  Restart Nodes
                </Button>
              </>
            )}
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
      <Dialog
        open={restartDialogOpen}
        onOpenChange={(open) => setRestartDialogOpen(open)}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Are you absolutely sure?</DialogTitle>
            <DialogDescription>
              This will restart the ROS 2 nodes, and may take a few minutes for
              it to come back up. You can also choose to reboot the host
              machine, which will take longer.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {onRestartNodes && (
              <DialogClose asChild>
                <Button variant="destructive" onClick={onRestartNodes}>
                  Restart Nodes
                </Button>
              </DialogClose>
            )}
            {onRebootSystem && (
              <DialogClose asChild>
                <Button variant="destructive" onClick={onRebootSystem}>
                  Reboot System
                </Button>
              </DialogClose>
            )}
            <DialogClose asChild>
              <Button>Cancel</Button>
            </DialogClose>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
