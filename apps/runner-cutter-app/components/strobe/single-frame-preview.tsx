"use client";

import useVideoServerStreamUrl from "@/lib/useVideoServerStreamUrl";
import { cn } from "@/lib/utils";
import { useEffect, useRef } from "react";

function uint8ArrayToBase64(byteArray: Uint8Array): string {
  let binary = "";
  for (let i = 0; i < byteArray.length; i++) {
    binary += String.fromCharCode(byteArray[i]);
  }
  return btoa(binary); // Encode binary data to Base64
}

function stringToUint8Array(str: string): Uint8Array {
  const byteArray = new Uint8Array(str.length);
  for (let i = 0; i < str.length; i++) {
    byteArray[i] = str.charCodeAt(i) & 0xff;
  }
  return byteArray;
}

function findUint8ArrayIndex(
  array: Uint8Array,
  searchArray: Uint8Array
): number {
  // Search for the index of the first occurrence of `searchArray` in `array`
  for (let i = 0; i <= array.length - searchArray.length; i++) {
    let found = true;
    for (let j = 0; j < searchArray.length; j++) {
      if (array[i + j] !== searchArray[j]) {
        found = false;
        break;
      }
    }
    if (found) {
      return i;
    }
  }
  return -1;
}

function extractJPEGData(data: Uint8Array): Uint8Array | null {
  const soiMarker = new Uint8Array([0xff, 0xd8]); // Start of Image marker (0xFFD8)
  const eoiMarker = new Uint8Array([0xff, 0xd9]); // End of Image marker (0xFFD9)

  // Find the index of the SOI marker
  const soiIndex = findUint8ArrayIndex(data, soiMarker);
  if (soiIndex === -1) {
    return null;
  }

  // Find the index of the EOI marker
  const eoiIndex = findUint8ArrayIndex(data, eoiMarker);
  if (eoiIndex === -1 || eoiIndex < soiIndex) {
    return null;
  }

  // Extract the valid JPEG data between SOI and EOI
  return data.slice(soiIndex, eoiIndex + eoiMarker.length);
}

export default function SingleFramePreview({
  topicName,
  quality = 30,
  enableStream = false,
  className,
}: {
  topicName?: string;
  quality?: number;
  enableStream?: boolean;
  className?: string;
}) {
  const imgRef = useRef<HTMLImageElement>(null);

  const streamUrl = useVideoServerStreamUrl(topicName, quality, enableStream);

  useEffect(() => {
    const abortController = new AbortController();

    async function fetchStream() {
      try {
        if (!streamUrl) {
          return;
        }

        const response = await fetch(streamUrl, {
          signal: abortController.signal,
        });
        if (!response.body) {
          return;
        }

        const reader = response.body.getReader();
        let partialData = new Uint8Array(0);
        const boundary = stringToUint8Array("--boundarydonotcross\r\n");

        while (true) {
          const { value, done } = await reader.read();
          if (done) {
            break;
          }

          // Append the new chunk to the buffer
          const newBuffer = new Uint8Array(partialData.length + value.length);
          newBuffer.set(partialData);
          newBuffer.set(value, partialData.length);
          partialData = newBuffer;

          // Look for the boundary in the buffer
          const boundaryIndex = findUint8ArrayIndex(value, boundary);
          if (boundaryIndex === -1) {
            continue;
          }

          // Extract everything before the boundary
          const partData = partialData.slice(0, boundaryIndex);
          // Keep remainder after the boundary
          // TODO: for some reason when there is data left over, we can't extract JPEG from the next frame.
          // partialData = partialData.slice(boundaryIndex + boundary.length);
          partialData = new Uint8Array(0);

          const jpegData = extractJPEGData(partData);
          if (!jpegData) {
            continue;
          }

          const imgElement = imgRef.current;
          if (!imgElement) {
            return;
          }
          imgElement.src = `data:image/jpeg;base64,${uint8ArrayToBase64(
            jpegData
          )}`;
        }
      } catch (error) {
        if (error instanceof Error && error.name !== "AbortError") {
          console.error("Stream error:", error);
        }
      }
    }

    fetchStream();

    return () => {
      abortController.abort();
    };
  }, [streamUrl]);

  return (
    <img
      ref={imgRef}
      alt="Camera Frame Preview"
      className={cn("object-contain bg-black", className)}
    />
  );
}
