"use client";

import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";

// Based on https://github.com/farm-ng/amiga-app-template/blob/main/ts/src/components/ExitButton.tsx
export default function ExitButton({ className }: { className?: string }) {
  const [appData, setAppData] = useState<{ [key: string]: any }>({});

  const handleClick = () => {
    const baseEndpoint = `http://${window.location.hostname}:8001/systemctl_action/`;

    const requestBody = {
      account_name: appData.account,
      service_id: appData.name,
      action: "stop",
      app_route: appData.app_route,
    };

    // request server start the service
    fetch(baseEndpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(requestBody),
    })
      .then((response) => response.json())
      .then((result) => {
        console.log("Service action response:", result);
        // redirect
        window.location.href = `${window.location.protocol}//${window.location.hostname}/apps/launcher`;
      })
      .catch((error) => {
        console.error("Error when clicking exit button:", error);
      });
  };

  useEffect(() => {
    const baseEndpoint = `http://${window.location.hostname}:8001/custom_app_info/${window.location.port}`;

    fetch(baseEndpoint)
      .then((response) => response.json())
      .then((result) => {
        if (result) {
          setAppData(result.service);
        }
      })
      .catch((error) => {
        console.warn("Could not fetch custom app info:", error);
      });
  }, []);

  return (
    appData.name && (
      <Button className={className} onClick={handleClick}>
        Quit
      </Button>
    )
  );
}
