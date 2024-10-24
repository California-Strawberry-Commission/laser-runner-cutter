import {
  Carousel,
  CarouselContent,
  CarouselItem,
} from "@/components/ui/carousel";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import type { NodeInfo } from "@/lib/NodeInfo";
import { cn } from "@/lib/utils";

function NodeCard({ nodeInfo }: { nodeInfo: NodeInfo }) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <div
          className={cn(
            "w-full rounded-lg px-4 py-2 cursor-pointer",
            nodeInfo.connected ? "bg-green-500" : "bg-red-500"
          )}
        >
          <p className="font-semibold text-xs overflow-hidden text-ellipsis">
            {nodeInfo.name}
          </p>
          <p className="text-xs overflow-hidden text-ellipsis">
            {nodeInfo.connected ? "Connected" : "Disconnected"}
          </p>
        </div>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>{nodeInfo.name}</DialogTitle>
          <DialogDescription>
            {nodeInfo.connected ? "Connected" : "Disconnected"}
          </DialogDescription>
        </DialogHeader>
        <pre className="overflow-hidden text-ellipsis text-xs">
          {JSON.stringify(nodeInfo.state ?? {}, undefined, 2)}
        </pre>
      </DialogContent>
    </Dialog>
  );
}

export default function NodesCarousel({
  nodeInfos,
  className,
  children,
}: {
  nodeInfos: NodeInfo[];
  className?: string;
  children?: React.ReactNode;
}) {
  return (
    <Carousel
      className={className}
      opts={{
        align: "start",
        dragFree: true,
      }}
    >
      <CarouselContent>
        {nodeInfos.map((nodeInfo) => (
          <CarouselItem className="basis-40" key={nodeInfo.name}>
            <NodeCard nodeInfo={nodeInfo} />
          </CarouselItem>
        ))}
        {children && (
          <CarouselItem className="basis-40" key="children">
            {children}
          </CarouselItem>
        )}
      </CarouselContent>
    </Carousel>
  );
}
