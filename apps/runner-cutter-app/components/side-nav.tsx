"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";

type SideNavItem = {
  title: string;
  path: string;
  icon?: JSX.Element;
};

const SIDENAV_ITEMS: SideNavItem[] = [
  {
    title: "Runner Cutter",
    path: "/",
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
    title: "Nodes",
    path: "/nodes",
  },
  {
    title: "Wi-Fi",
    path: "/wifi",
  },
];

export default function SideNav() {
  const pathname = usePathname();

  return (
    <div className="flex flex-col gap-4 w-full">
      {SIDENAV_ITEMS.map((item, idx) => {
        return (
          <Link key={idx} href={item.path}>
            <Button className="w-full" disabled={item.path === pathname}>
              {item.icon} {item.title}
            </Button>
          </Link>
        );
      })}
    </div>
  );
}
