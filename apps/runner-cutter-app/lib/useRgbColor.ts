import { useCallback, useState } from "react";

export type RgbColor = { r: number; g: number; b: number };

function shallowEqualRGB(a: RgbColor, b: RgbColor, epsilon = 1e-5): boolean {
  return (
    Math.abs(a.r - b.r) < epsilon &&
    Math.abs(a.g - b.g) < epsilon &&
    Math.abs(a.b - b.b) < epsilon
  );
}

export default function useRgbColor(initial: RgbColor) {
  const [color, setColorInternal] = useState<RgbColor>(initial);

  const setColor = useCallback((next: RgbColor) => {
    setColorInternal((prev) => {
      return shallowEqualRGB(prev, next) ? prev : next;
    });
  }, []);

  return [color, setColor] as const;
}
