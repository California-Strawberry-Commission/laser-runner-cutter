"use client";

import { useEffect, useState } from "react";
import useROS from "@/lib/ros/useROS";
import { Button } from "@/components/ui/button";

const LASER_STATES = ["disconnected", "stopped", "playing"];

export default function Controls() {
  const { ros, getNodes, callService, subscribe } = useROS();
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  const [nodes, setNodes] = useState<string[]>([]);
  const [laserNodeName, setLaserNodeName] = useState<string>("/laser0");
  const [laserState, setLaserState] = useState<number>(0);

  useEffect(() => {
    const loadNodes = async () => {
      const nodes = await getNodes();
      setNodes(nodes);
    };
    loadNodes();

    setRosConnected(ros.isConnected);
  }, []);

  useEffect(() => {
    const getState = async () => {
      const state = await callService(
        `${laserNodeName}/get_state`,
        "laser_control_interfaces/GetState",
        {}
      );
      setLaserState(state.state.data);
    };
    getState();
  }, [laserNodeName]);

  // Subscriptions
  useEffect(() => {
    const playingSub = subscribe(
      `${laserNodeName}/playing`,
      "std_msgs/Bool",
      (message) => {
        const playing = message.data;
        setLaserState(message.data ? 2 : 1);
      }
    );
    return () => {
      playingSub.unsubscribe();
    };
  }, [laserNodeName]);

  const onPlayClick = () => {
    callService(`${laserNodeName}/play`, "std_srvs/Empty", {});
  };

  const onStopClick = () => {
    callService(`${laserNodeName}/stop`, "std_srvs/Empty", {});
  };

  const onRandomizeColorClick = () => {
    callService(
      `${laserNodeName}/set_color`,
      "laser_control_interfaces/SetColor",
      {
        r: Math.random(),
        g: Math.random(),
        b: Math.random(),
        i: 0.0,
      }
    );
  };

  return (
    <div className="flex flex-col gap-4 items-center">
      <div className="flex flex-col items-center">
        <p className="text-center">{`Rosbridge: ${
          rosConnected ? "connected" : "disconnected"
        }`}</p>
        <p className="text-center">{`Laser (${laserNodeName}): ${LASER_STATES[laserState]}`}</p>
      </div>
      <div className="flex flex-row gap-4">
        <Button disabled={!rosConnected} onClick={onPlayClick}>
          Play
        </Button>
        <Button disabled={!rosConnected} onClick={onStopClick}>
          Stop
        </Button>
        <Button disabled={!rosConnected} onClick={onRandomizeColorClick}>
          Randomize Color
        </Button>
      </div>
    </div>
  );
}
