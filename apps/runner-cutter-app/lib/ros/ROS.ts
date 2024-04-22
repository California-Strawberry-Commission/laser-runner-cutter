import ROSLIB from "roslib";

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
  private url: string;
  private ros: ROSLIB.Ros;
  private reconnectIntervalMs: number;
  private reconnectInterval: NodeJS.Timeout | null;
  private nodes: string[];
  private nodeMonitorInterval: NodeJS.Timeout | null;
  private nodeListeners: ((nodeName: string, connected: boolean) => void)[];

  constructor(url: string, reconnectIntervalMs: number = 1000) {
    this.url = url;
    this.reconnectIntervalMs = reconnectIntervalMs;
    this.reconnectInterval = null;

    this.ros = new ROSLIB.Ros({ url });
    this.ros.on("connection", () => {
      console.log("[ROS] Connected");
      // Stop reconnect, start node monitor
      if (this.reconnectInterval !== null) {
        clearInterval(this.reconnectInterval);
        this.reconnectInterval = null;
      }
      this.setupNodeMonitor();
    });
    this.ros.on("close", () => {
      console.log("[ROS] Disconnected");
      // Start reconnect, stop node monitor
      this.setupReconnect();
      if (this.nodeMonitorInterval !== null) {
        clearInterval(this.nodeMonitorInterval);
        this.nodeMonitorInterval = null;
      }
    });
    this.ros.on("error", () => {
      console.log("[ROS] Error connecting");
      // Start reconnect, stop node monitor
      this.setupReconnect();
      if (this.nodeMonitorInterval !== null) {
        clearInterval(this.nodeMonitorInterval);
        this.nodeMonitorInterval = null;
      }
    });

    this.nodes = [];
    this.nodeListeners = [];
    this.nodeMonitorInterval = null;
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
    values: any
  ): Promise<any> {
    const client = new ROSLIB.Service({ ros: this.ros, name, serviceType });
    const request = new ROSLIB.ServiceRequest(values);
    return new Promise<any>((resolve, reject) => {
      client.callService(
        request,
        (response) => {
          resolve(response);
        },
        (error) => {
          reject(error);
        }
      );
    });
  }

  subscribe(
    name: string,
    messageType: string,
    callback: (message: any) => void
  ): ROSLIB.Topic<ROSLIB.Message> {
    const listener = new ROSLIB.Topic({ ros: this.ros, name, messageType });
    listener.subscribe(callback);
    return listener;
  }

  private async getNodesInternal(): Promise<string[]> {
    const result = await this.callService("/rosapi/nodes", "rosapi/Nodes", {});
    return result.nodes;
  }

  private setupReconnect(): void {
    if (this.reconnectInterval === null) {
      this.reconnectInterval = setInterval(() => {
        console.log("[ROS] Attempting to reconnect to rosbridge...");
        this.ros.connect(this.url);
      }, this.reconnectIntervalMs);
    }
  }

  private setupNodeMonitor(): void {
    if (this.nodeMonitorInterval === null) {
      this.nodeMonitorInterval = setInterval(async () => {
        if (!this.isConnected()) {
          return;
        }

        const nodes = await this.getNodesInternal();
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
      }, 1000);
    }
  }
}
