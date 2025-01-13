"use client";

import ExitButton from "@/components/exit-button";
import { Button } from "@/components/ui/button";
import useNotifications from "@/lib/useNotifications";
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
