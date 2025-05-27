import ROSLIB from "roslib";
import TaskRunner from "@/lib/ros/TaskRunner";

class EventSubscriptionHandle {
  private onUnsubscribe: () => void;

  constructor(onUnsubscribe: () => void) {
    this.onUnsubscribe = onUnsubscribe;
  }

  unsubscribe() {
    this.onUnsubscribe();
  }
}

export default class ROS {
  private url: string | null;
  private ros: ROSLIB.Ros;
  private reconnectIntervalMs: number;
  private reconnectRunner: TaskRunner | null = null;
  private nodes: string[] = [];
  private nodeMonitorIntervalMs: number;
  private nodeMonitorRunner: TaskRunner | null = null;
  private nodeListeners: ((nodeName: string, connected: boolean) => void)[] =
    [];
  private serviceCache: Map<
    string,
    ROSLIB.Service<ROSLIB.ServiceRequest, ROSLIB.ServiceResponse>
  > = new Map();
  private topicCache: Map<string, ROSLIB.Topic<ROSLIB.Message>> = new Map();

  constructor(
    url: string | null,
    reconnectIntervalMs: number = 5000,
    nodeMonitorIntervalMs: number = 1000
  ) {
    this.url = url;
    this.reconnectIntervalMs = reconnectIntervalMs;
    this.nodeMonitorIntervalMs = nodeMonitorIntervalMs;

    this.ros = new ROSLIB.Ros(url ? { url } : {});
    this.ros.on("connection", () => {
      console.log("[ROS] Connected");
      this.onRosConnected();
    });
    this.ros.on("close", () => {
      console.log("[ROS] Disconnected");
      this.onRosDisconnected();
    });
    this.ros.on("error", () => {
      console.log("[ROS] Error connecting");
      this.onRosDisconnected();
    });
  }

  isConnected(): boolean {
    return this.ros.isConnected;
  }

  onStateChange(callback: (state: string) => void): EventSubscriptionHandle {
    const connectionCallback = () => callback("connection");
    const errorCallback = () => callback("error");
    const closeCallback = () => callback("close");
    this.ros.on("connection", connectionCallback);
    this.ros.on("error", errorCallback);
    this.ros.on("close", closeCallback);
    return new EventSubscriptionHandle(() => {
      this.ros.off("connection", connectionCallback);
      this.ros.off("error", errorCallback);
      this.ros.off("close", closeCallback);
    });
  }

  onNodeConnected(
    callback: (nodeName: string, connected: boolean) => void
  ): EventSubscriptionHandle {
    this.nodeListeners.push(callback);
    return new EventSubscriptionHandle(() => {
      const index = this.nodeListeners.indexOf(callback);
      if (index > -1) {
        this.nodeListeners.splice(index, 1);
      }
    });
  }

  getNodes(): string[] {
    return this.nodes;
  }

  isNodeConnected(nodeName: string): boolean {
    return this.nodes.includes(nodeName);
  }

  async callService(
    name: string,
    serviceType: string,
    values: any,
    timeoutMs: number = 0
  ): Promise<any> {
    let client = this.serviceCache.get(name);
    if (!client) {
      client = new ROSLIB.Service({ ros: this.ros, name, serviceType });
      this.serviceCache.set(name, client);
    }

    const request = new ROSLIB.ServiceRequest(values);
    return new Promise<any>((resolve, reject) => {
      let timeoutId: any;

      client.callService(
        request,
        (response) => {
          clearTimeout(timeoutId);
          resolve(response);
        },
        (error) => {
          clearTimeout(timeoutId);
          reject(error);
        }
      );

      if (timeoutMs > 0) {
        timeoutId = setTimeout(() => {
          reject(new Error(`Service call timed out after ${timeoutMs} ms`));
        }, timeoutMs);
      }
    });
  }

  subscribe(
    name: string,
    messageType: string,
    callback: (message: ROSLIB.Message) => void
  ): ROSLIB.Topic<ROSLIB.Message> {
    let topic = this.topicCache.get(name);
    if (!topic) {
      topic = new ROSLIB.Topic({ ros: this.ros, name, messageType });
      this.topicCache.set(name, topic);
    }
    topic.subscribe(callback);
    return topic;
  }

  publish(
    name: string,
    messageType: string,
    message: ROSLIB.Message
  ): ROSLIB.Topic<ROSLIB.Message> {
    let topic = this.topicCache.get(name);
    if (!topic) {
      topic = new ROSLIB.Topic({ ros: this.ros, name, messageType });
      this.topicCache.set(name, topic);
    }
    topic.publish(message);
    return topic;
  }

  private async getNodesInternal(): Promise<string[]> {
    const result = await this.callService(
      "/rosapi/nodes",
      "rosapi/Nodes",
      {},
      1000
    );
    return result.nodes;
  }

  private onRosConnected(): void {
    // Stop reconnect, start node monitor
    if (this.reconnectRunner !== null) {
      this.reconnectRunner.stop();
      this.reconnectRunner = null;
    }
    if (this.nodeMonitorRunner === null) {
      this.nodeMonitorRunner = new TaskRunner(
        this.nodeMonitorTask.bind(this),
        this.nodeMonitorIntervalMs
      );
      this.nodeMonitorRunner.start();
    }
  }

  private onRosDisconnected(): void {
    // Start reconnect, stop node monitor
    if (this.reconnectRunner === null) {
      this.reconnectRunner = new TaskRunner(
        this.reconnectTask.bind(this),
        this.reconnectIntervalMs
      );
      this.reconnectRunner.start();
    }
    if (this.nodeMonitorRunner !== null) {
      this.nodeMonitorRunner.stop();
      this.nodeMonitorRunner = null;
    }
  }

  private reconnectTask(): void {
    console.log(`[ROS] Attempting to reconnect to rosbridge at ${this.url}...`);
    if (this.url) {
      this.ros.connect(this.url);
    }
  }

  private async nodeMonitorTask(): Promise<void> {
    if (!this.isConnected()) {
      return;
    }

    let nodes: string[] = [];
    try {
      nodes = await this.getNodesInternal();
    } catch (error) {
      console.error("[ROS] Failed to get nodes:", error);
    }
    const prevSet = new Set(this.nodes);
    const currSet = new Set(nodes);
    const disconnected = this.nodes.filter((node) => !currSet.has(node));
    const connected = nodes.filter((node) => !prevSet.has(node));
    this.nodes = nodes;

    disconnected.forEach((node) => {
      console.log(`[ROS] Node disconnected: ${node}`);
      this.nodeListeners.forEach((listener) => {
        listener(node, false);
      });
    });

    connected.forEach((node) => {
      console.log(`[ROS] Node connected: ${node}`);
      this.nodeListeners.forEach((listener) => {
        listener(node, true);
      });
    });
  }
}
