import { useContext } from "react";
import ROSLIB from "roslib";
import ROSContext from "@/lib/ros/ROSContext";

export default function useROS() {
  const ros = useContext(ROSContext);

  // TODO: add listener for rosbridge connection state

  async function getNodes(): Promise<string[]> {
    return new Promise<string[]>((resolve, reject) => {
      ros.getNodes(
        (nodes) => {
          resolve(nodes);
        },
        (error) => {
          reject(error);
        }
      );
    });
  }

  async function callService(
    name: string,
    serviceType: string,
    values: any
  ): Promise<any> {
    const client = new ROSLIB.Service({ ros, name, serviceType });
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

  function subscribe(
    name: string,
    messageType: string,
    callback: (message: any) => void
  ) {
    const listener = new ROSLIB.Topic({ ros, name, messageType });
    listener.subscribe(callback);
    return listener;
  }

  return { ros, getNodes, callService, subscribe };
}
