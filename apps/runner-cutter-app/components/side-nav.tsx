"use client";

import ExitButton from "@/components/exit-button";
import { Button } from "@/components/ui/button";
import useNotifications from "@/lib/useNotifications";
import Link from "next/link";
import { usePathname } from "next/navigation";

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
    title: "Aim Test",
    path: "/test/aim",
  },
  {
    title: "Track Test",
    path: "/test/track",
  },
  {
    title: "Navigation",
    path: "/test/navigation",
  },
  {
    title: "Wi-Fi",
    path: "/wifi",
  },
];

export default function SideNav() {
  const pathname = usePathname();
  useNotifications("/notifications");

  return (
    <div className="flex flex-col gap-4 w-full h-full">
      {SIDENAV_ITEMS.map((item, idx) => {
        return (
          <Link key={idx} href={item.path}>
            <Button className="w-full" disabled={item.path === pathname}>
              {item.icon} {item.title}
            </Button>
          </Link>
        );
      })}
      <div className="mt-auto">
        <ExitButton className="w-full" />
      </div>
    </div>
  );
}
