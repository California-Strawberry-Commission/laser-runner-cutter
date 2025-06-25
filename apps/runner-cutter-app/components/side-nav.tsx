"use client";

import ExitButton from "@/components/exit-button";
import { Button } from "@/components/ui/button";
import useNotifications from "@/lib/useNotifications";
import { usePathname } from "next/navigation";
import { useState } from "react";

type SideNavItem = {
  title: string;
  path: string;
  isTestPage: boolean;
  icon?: React.ReactElement;
};

const SIDENAV_ITEMS: SideNavItem[] = [
  {
    title: "Runner Cutter",
    path: "/",
    isTestPage: false,
  },
  {
    title: "Settings",
    path: "/settings",
    isTestPage: false,
  },
  {
    title: "Camera Test",
    path: "/test/camera",
    isTestPage: true,
  },
  {
    title: "Laser Test",
    path: "/test/laser",
    isTestPage: true,
  },
  {
    title: "Calibration Test",
    path: "/test/calibration",
    isTestPage: true,
  },
  {
    title: "Track Test",
    path: "/test/track",
    isTestPage: true,
  },
  {
    title: "Strobe Test",
    path: "/test/strobe",
    isTestPage: true,
  },
  {
    title: "Navigation",
    path: "/test/navigation",
    isTestPage: true,
  },
  {
    title: "Wi-Fi",
    path: "/wifi",
    isTestPage: false,
  },
];

export default function SideNav() {
  const pathname = usePathname();
  useNotifications("/notifications");

  const [showTestPages, setShowTestPages] = useState<boolean>(false);

  return (
    <div className="flex flex-col gap-4 w-full h-full">
      {SIDENAV_ITEMS.map((item, idx) => {
        return (
          (!item.isTestPage || showTestPages) && (
            // TODO: Unfortunately we don't use <Link> here because it is causing issues with the
            // camera preview. When using <Link>, existing connections to web_video_server are
            // not closed, and eventually we are not able to create a new connection (and thus the
            // stream fails to render)
            <a key={idx} href={item.path}>
              <Button className="w-full" disabled={item.path === pathname}>
                {item.icon} {item.title}
              </Button>
            </a>
          )
        );
      })}
      <div className="mt-auto flex flex-col gap-4 w-full">
        <Button
          className="w-full"
          onClick={() => {
            setShowTestPages(!showTestPages);
          }}
        >
          {showTestPages ? "Hide Test Pages" : "Show Test Pages"}
        </Button>
        <ExitButton className="w-full" />
      </div>
    </div>
  );
}
