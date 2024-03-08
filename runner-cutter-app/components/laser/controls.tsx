"use client";

import { useEffect, useState } from "react";
import useROS from "@/lib/ros/useROS";
import { Button } from "@/components/ui/button";

const LASER_STATES = ["disconnected", "stopped", "playing"];

export default function Controls() {
  const { ros, callService, subscribe } = useROS();
  const [rosConnected, setRosConnected] = useState<boolean>(false);
  // TODO: add ability to select laser node name
  const [laserNodeName, setLaserNodeName] = useState<string>("/laser0");
  const [laserState, setLaserState] = useState<number>(0);

  useEffect(() => {
    // TODO: add listener for rosbridge connection state
    setRosConnected(ros.isConnected);
  }, []);

  // Initial state
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
    const stateSub = subscribe(
      `${laserNodeName}/state`,
      "laser_control_interfaces/State",
      (message) => {
        setLaserState(message.data);
      }
    );
    return () => {
      stateSub.unsubscribe();
    };
  }, [laserNodeName]);

  const onAddRandomPointClick = () => {
    callService(
      `${laserNodeName}/add_point`,
      "laser_control_interfaces/AddPoint",
      {
        // TODO: normalize AddPoint x, y to [0, 1]. Just assume Helios for now
        point: {
          x: Math.floor(Math.random() * 4096),
          y: Math.floor(Math.random() * 4096),
        },
      }
    );
  };

  const onClearPointsClick = () => {
    callService(`${laserNodeName}/clear_points`, "std_srvs/Empty", {});
  };

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
        <Button disabled={!rosConnected} onClick={onAddRandomPointClick}>
          Add Random Point
        </Button>
        <Button disabled={!rosConnected} onClick={onClearPointsClick}>
          Clear Points
        </Button>
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
