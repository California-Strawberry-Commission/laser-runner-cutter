"use client";

import { usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";
import ExitButton from "@/components/exit-button";

type SideNavItem = {
  title: string;
  path: string;
  icon?: JSX.Element;
};

const SIDENAV_ITEMS: SideNavItem[] = [
  {
    title: "Setup",
    path: "/setup",
  },
  {
    title: "Runner Cutter",
    path: "/runner-cutter",
  },
  {
    title: "Camera Test",
    path: "/test/camera",
  },
  {
    title: "Laser Test",
    path: "/test/laser",
  },
  {
    title: "Calibration Test",
    path: "/test/calibration",
  },
  {
    title: "Aim Test",
    path: "/test/aim",
  },
  {
    title: "Wi-Fi",
    path: "/wifi",
  },
  {
    title: "Navigation",
    path: "/test/navigation",
  },
];

export default function SideNav() {
  const pathname = usePathname();

  return (
    <div className="flex flex-col gap-4 w-full h-full">
      {SIDENAV_ITEMS.map((item, idx) => {
        return (
          // TODO: Unfortunately we don't use <Link> here because it is causing issues with the
          // camera preview. When using <Link>, existing connections to web_video_server are
          // not closed, and eventually we are not able to create a new connection (and thus the
          // stream fails to render)
          <a key={idx} href={item.path}>
            <Button className="w-full" disabled={item.path === pathname}>
              {item.icon} {item.title}
            </Button>
          </a>
        );
      })}
      <div className="mt-auto">
        <ExitButton className="w-full" />
      </div>
    </div>
  );
}
