import ROSLIB from "roslib";

export default class ROS {
  url: string;
  ros: ROSLIB.Ros;
  reconnectIntervalMs: number;
  reconnectInterval: NodeJS.Timeout | null;

  constructor(url: string, reconnectIntervalMs: number = 1000) {
    this.url = url;
    this.reconnectIntervalMs = reconnectIntervalMs;
    this.reconnectInterval = null;

    this.ros = new ROSLIB.Ros({ url });
    this.ros.on("connection", () => {
      console.log("[ROS] Connected");
      if (this.reconnectInterval !== null) {
        clearInterval(this.reconnectInterval);
        this.reconnectInterval = null;
      }
    });
    this.ros.on("close", () => {
      console.log("[ROS] Disconnected");
      this.setupReconnect();
    });
    this.ros.on("error", () => {
      console.log("[ROS] Error connecting");
      this.setupReconnect();
    });
  }

  private setupReconnect(): void {
    if (this.reconnectInterval === null) {
      this.reconnectInterval = setInterval(() => {
        console.log("[ROS] Attempting to reconnect to rosbridge...");
        this.ros.connect(this.url);
      }, this.reconnectIntervalMs);
    }
  }

  onStateChange(callback: (state: string) => void) {
    this.ros.on("connection", () => callback("connection"));
    this.ros.on("error", () => callback("error"));
    this.ros.on("close", () => callback("close"));
  }

  async getNodes(): Promise<string[]> {
    return new Promise<string[]>((resolve, reject) => {
      this.ros.getNodes(
        (nodes) => {
          resolve(nodes);
        },
        (error) => {
          reject(error);
        }
      );
    });
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
  ) {
    const listener = new ROSLIB.Topic({ ros: this.ros, name, messageType });
    listener.subscribe(callback);
    return listener;
  }
}
