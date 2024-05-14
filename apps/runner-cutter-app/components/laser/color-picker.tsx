"use client";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { cn } from "@/lib/utils";
import { Paintbrush } from "lucide-react";

export default function ColorPicker({
  color,
  onColorChange,
  className,
}: {
  color: string;
  onColorChange: (color: string) => void;
  className?: string;
}) {
  const solids = [
    "#ff0000",
    "#00ff00",
    "#0000ff",
    "#7f007f",
    "#7f7f00",
    "#007f7f",
    "#555555",
    "#000000",
  ];

  return (
    <Popover>
      <PopoverTrigger asChild>
        <Button
          variant={"outline"}
          className={cn(
            "w-[220px] justify-start text-left font-normal",
            !color && "text-muted-foreground",
            className
          )}
        >
          <div className="flex w-full items-center gap-2">
            {color ? (
              <div
                className="h-4 w-4 rounded !bg-cover !bg-center transition-all"
                style={{ background: color }}
              ></div>
            ) : (
              <Paintbrush className="h-4 w-4" />
            )}
            <div className="flex-1 truncate">
              {color ? color : "Pick a color"}
            </div>
          </div>
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-64">
        <div className="mt-0 flex flex-wrap gap-1">
          {solids.map((s) => (
            <div
              key={s}
              style={{ background: s }}
              className="h-6 w-6 cursor-pointer rounded-md active:scale-105"
              onClick={() => onColorChange(s)}
            />
          ))}
        </div>

        <Input
          id="custom"
          value={color}
          className="col-span-2 mt-4 h-8"
          onChange={(str) => onColorChange(str)}
        />
      </PopoverContent>
    </Popover>
  );
}
