"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";

export default function PasswordModal({
  networkName,
  onSubmit,
}: {
  networkName: string;
  onSubmit?: (networkName: string, password: string) => void;
}) {
  const [password, setPassword] = useState<string>("");

  return (
    <DialogContent className="sm:max-w-[425px]">
      <DialogHeader>
        <DialogTitle>Enter Wi-Fi password</DialogTitle>
        <DialogDescription>
          Provide the password for the Wi-Fi network "{networkName}":
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
            onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
              setPassword(e.target.value);
            }}
          />
        </div>
      </div>
      <DialogFooter>
        <DialogClose asChild>
          <Button
            type="submit"
            onClick={() => {
              onSubmit && onSubmit(networkName, password);
            }}
          >
            Connect
          </Button>
        </DialogClose>
      </DialogFooter>
    </DialogContent>
  );
}
