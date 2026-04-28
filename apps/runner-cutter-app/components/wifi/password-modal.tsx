"use client";

import { Button } from "@/components/ui/button";
import {
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useState } from "react";

export default function PasswordModal({
  networkName,
  onSubmit,
  error = false,
}: {
  networkName: string;
  onSubmit?: (networkName: string, password: string) => void;
  error?: boolean;
}) {
  const [password, setPassword] = useState("");

  function handleSubmit() {
    if (onSubmit) {
      onSubmit(networkName, password);
    }
    setPassword("");
  }

  return (
    <DialogContent className="sm:max-w-[425px]">
      <DialogHeader>
        <DialogTitle>
          {error ? "Error connecting to Wi-Fi network" : "Enter Wi-Fi password"}
        </DialogTitle>
        <DialogDescription>
          {error
            ? `There was an error when attempting to connect to the Wi-Fi network "${networkName}". Please try again:`
            : `Provide the password for the Wi-Fi network "${networkName}":`}
        </DialogDescription>
      </DialogHeader>
      <div className="grid gap-4 py-4">
        <div className="grid grid-cols-4 items-center gap-4">
          <Label htmlFor="password" className="text-right">
            Password
          </Label>
          <Input
            id="password"
            type="password"
            className="col-span-3"
            value={password}
            onChange={setPassword}
          />
        </div>
      </div>
      <DialogFooter>
        <DialogClose asChild>
          <Button type="button" onClick={handleSubmit}>
            Connect
          </Button>
        </DialogClose>
      </DialogFooter>
    </DialogContent>
  );
}
