import type { NodeInfo } from "@/lib/NodeInfo";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
} from "@/components/ui/carousel";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";

function NodeCard({ nodeInfo }: { nodeInfo: NodeInfo }) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Card
          className={cn(
            "max-w-48",
            nodeInfo.connected ? "bg-green-500" : "bg-red-500"
          )}
        >
          <CardHeader className="p-4">
            <CardTitle className="text-lg overflow-hidden text-ellipsis">
              {nodeInfo.name}
            </CardTitle>
            <CardDescription className="text-foreground">
              {nodeInfo.connected ? "Connected" : "Disconnected"}
            </CardDescription>
          </CardHeader>
        </Card>
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
          <CarouselItem className="basis-1/4" key={nodeInfo.name}>
            <NodeCard nodeInfo={nodeInfo} />
          </CarouselItem>
        ))}
        {children && (
          <CarouselItem className="basis-1/4" key="children">
            {children}
          </CarouselItem>
        )}
      </CarouselContent>
      <CarouselPrevious />
      <CarouselNext />
    </Carousel>
  );
}
