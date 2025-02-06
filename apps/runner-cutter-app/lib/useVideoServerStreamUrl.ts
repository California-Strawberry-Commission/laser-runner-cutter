import { useEffect, useState } from "react";

export default function useVideoServerStreamUrl(
  topicName: string = "",
  quality: number = 30,
  enableStream: boolean = false
) {
  const [streamUrl, setStreamUrl] = useState<string>("");

  // Unfortunately, with SSR, this needed for code that should only run on client side. Otherwise
  // we will get an error when enableStream is true on initial render since the server and
  // client "src" values will not match.
  useEffect(() => {
    if (typeof window !== "undefined" && enableStream && topicName) {
      const videoServer =
        process.env.NEXT_PUBLIC_VIDEO_SERVER_URL ??
        `http://${window.location.hostname}:8080`;
      setStreamUrl(
        `${videoServer}/stream?topic=${topicName}&quality=${quality}&qos_profile=sensor_data`
      );
    } else {
      setStreamUrl("");
    }
  }, [topicName, quality, enableStream]);

  return streamUrl;
}
