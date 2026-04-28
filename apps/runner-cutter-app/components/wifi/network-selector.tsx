"use client";

import { Dialog } from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import PasswordModal from "@/components/wifi/password-modal";
import { Check } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

export type WifiNetwork = {
  ssid: string;
  signal: number;
  security: string;
  connected: boolean;
};

export default function NetworkSelector({
  fetchIntervalMs = 10000,
}: {
  fetchIntervalMs?: number;
}) {
  const [networks, setNetworks] = useState<WifiNetwork[]>([]);
  const [selectedSsid, setSelectedSsid] = useState<string | null>(null);
  const [passwordModalOpen, setPasswordModalOpen] = useState<boolean>(false);
  const [errorModalOpen, setErrorModalOpen] = useState<boolean>(false);

  const fetchNetworks = useCallback(async () => {
    const response = await fetch("/api/wifi/list");
    if (response.ok) {
      const data: WifiNetwork[] = await response.json();
      data.sort((a, b) => {
        // Sort by connected status (true first), followed by signal strength
        // in descending order
        if (a.connected !== b.connected) {
          return a.connected ? -1 : 1;
        }
        return b.signal - a.signal;
      });
      setNetworks(data);
    } else {
      console.error("Error fetching Wi-Fi networks");
    }
  }, []);

  useEffect(() => {
    fetchNetworks();
    const fetchNetworksInterval = setInterval(fetchNetworks, fetchIntervalMs);
    return () => clearInterval(fetchNetworksInterval);
  }, [fetchNetworks, fetchIntervalMs]);

  const onPasswordSubmit = useCallback(
    async (ssid: string, password: string) => {
      // TODO: show "connecting" modal
      const response = await fetch("/api/wifi/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ssid, password }),
      });

      if (response.ok) {
        fetchNetworks();
      } else {
        console.error("Error connecting to Wi-Fi network");
        setErrorModalOpen(true);
      }
    },
    [fetchNetworks],
  );

  return (
    <>
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
              className={!network.connected ? "cursor-pointer" : undefined}
              onClick={() => {
                if (!network.connected) {
                  setSelectedSsid(network.ssid);
                  setPasswordModalOpen(true);
                }
              }}
            >
              <TableCell className="pr-0">
                {network.connected && <Check className="h-4 w-4" />}
              </TableCell>
              <TableCell>{network.ssid}</TableCell>
              <TableCell>{network.signal}</TableCell>
              <TableCell>{network.security}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Dialog open={passwordModalOpen} onOpenChange={setPasswordModalOpen}>
        <PasswordModal
          networkName={selectedSsid ?? ""}
          onSubmit={onPasswordSubmit}
        />
      </Dialog>
      <Dialog open={errorModalOpen} onOpenChange={setErrorModalOpen}>
        <PasswordModal
          networkName={selectedSsid ?? ""}
          onSubmit={onPasswordSubmit}
          error
        />
      </Dialog>
    </>
  );
}
