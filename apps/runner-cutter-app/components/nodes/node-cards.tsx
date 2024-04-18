import type { NodeInfo } from "@/lib/NodeInfo";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

function NodeCard({ nodeInfo }: { nodeInfo: NodeInfo }) {
  return (
    <Card className={`${nodeInfo.connected ? "bg-green-500" : "bg-red-500"}`}>
      <CardHeader className="p-4">
        <CardTitle>{nodeInfo.name}</CardTitle>
        <CardDescription className="text-foreground">
          {nodeInfo.connected ? "Connected" : "Disconnected"}
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 pt-0 text-xs">
        <pre>{JSON.stringify(nodeInfo.state, undefined, 2)}</pre>
      </CardContent>
    </Card>
  );
}

export default function NodeCards({ nodeInfos }: { nodeInfos: NodeInfo[] }) {
  return (
    <div className="flex flex-row gap-4">
      {nodeInfos.map((nodeInfo) => (
        <NodeCard key={nodeInfo.name} nodeInfo={nodeInfo} />
      ))}
    </div>
  );
}
