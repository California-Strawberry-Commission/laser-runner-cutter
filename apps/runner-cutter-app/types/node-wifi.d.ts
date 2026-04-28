declare module "node-wifi" {
  export interface WiFiNetwork {
    ssid: string;
    bssid: string;
    mode: string;
    channel: number;
    frequency: number;
    signal_level: number;
    quality: number;
    security: string;
  }

  export function init(opts: { iface: string | null }): void;
  export function connect(opts: { ssid: string; password?: string }): Promise<void>;
  export function disconnect(): Promise<void>;
  export function scan(): Promise<WiFiNetwork[]>;
  export function getCurrentConnections(): Promise<WiFiNetwork[]>;
}
