"use client";

import { usePathname } from "next/navigation";
import Link from "next/link";
import { Button } from "@/components/ui/button";

export type SideNavItem = {
  title: string;
  path: string;
  icon?: JSX.Element;
};

export const SIDENAV_ITEMS: SideNavItem[] = [
  {
    title: "Laser Runner Cutter",
    path: "/",
  },
  {
    title: "Calibration",
    path: "/calibration",
  },
  {
    title: "Camera Test",
    path: "/camera",
  },
  {
    title: "Laser Test",
    path: "/laser",
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
