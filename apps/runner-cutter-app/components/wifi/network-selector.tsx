"use client";

import React, { useEffect, useState } from "react";
import { Check } from "lucide-react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Dialog } from "@/components/ui/dialog";
import PasswordModal from "@/components/wifi/password-modal";
import ErrorModal from "@/components/wifi/error-modal";

export type WifiNetwork = {
  ssid: string;
  signal: number;
  security: string;
  connected: boolean;
};

export default function NetworkSelector() {
  const [networks, setNetworks] = useState<WifiNetwork[]>([]);
  const [selectedSsid, setSelectedSsid] = useState<string>("");
  const [passwordModalOpen, setPasswordModalOpen] = useState<boolean>(false);
  const [errorModalOpen, setErrorModalOpen] = useState<boolean>(false);

  const fetchNetworks = async () => {
    const response = await fetch("/api/wifi/list");
    if (response.ok) {
      const data: WifiNetwork[] = await response.json();
      data.sort((a, b) => {
        // Sort by connected status (true first)
        if (a.connected !== b.connected) {
          return a.connected ? -1 : 1;
        }

        // If connected status is the same, sort by signal strength in descending order
        return b.signal - a.signal;
      });
      setNetworks(data);
    } else {
      console.error("Error fetching Wi-Fi networks");
    }
  };

  useEffect(() => {
    fetchNetworks();
    const fetchNetworksInterval = setInterval(fetchNetworks, 10000);
    return () => clearInterval(fetchNetworksInterval);
  }, []);

  const onPasswordSubmit = async (ssid: string, password: string) => {
    // TODO: show "connecting" modal
    const response = await fetch("/api/wifi/connect", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ ssid, password }),
    });

    if (response.ok) {
      console.log(`Successfully connected to network with SSID "${ssid}"`);
      fetchNetworks();
    } else {
      console.error("Error connecting to Wi-Fi network");
      setErrorModalOpen(true);
    }
  };

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead className="w-6"></TableHead>
          <TableHead>Name</TableHead>
          <TableHead className="w-24">Signal</TableHead>
          <TableHead className="w-36">Security</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {networks.map((network) => (
          <TableRow
            key={network.ssid}
            onClick={() => {
              if (!network.connected) {
                setSelectedSsid(network.ssid);
                setPasswordModalOpen(true);
              }
            }}
          >
            <TableCell className="pr-0">
              {network.connected ? <Check className="h-4 w-4" /> : null}
            </TableCell>
            <TableCell>{network.ssid}</TableCell>
            <TableCell>{network.signal}</TableCell>
            <TableCell>{network.security}</TableCell>
          </TableRow>
        ))}
        <Dialog open={passwordModalOpen} onOpenChange={setPasswordModalOpen}>
          <PasswordModal
            networkName={selectedSsid}
            onSubmit={onPasswordSubmit}
          />
        </Dialog>
        <Dialog open={errorModalOpen} onOpenChange={setErrorModalOpen}>
          <ErrorModal networkName={selectedSsid} onSubmit={onPasswordSubmit} />
        </Dialog>
      </TableBody>
    </Table>
  );
}
