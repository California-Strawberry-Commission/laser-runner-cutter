using System;
using UnityEngine;

public class TextureUtils
{
    public static sensor_msgs.msg.Image ConvertToImageMsg(Texture2D frame)
    {
        if (frame.format != TextureFormat.RGB24 && frame.format != TextureFormat.R16)
        {
            throw new ArgumentException($"Unsupported texture format: {frame.format}");
        }

        bool isColor = frame.format == TextureFormat.RGB24;
        uint width = unchecked((uint)frame.width);
        uint height = unchecked((uint)frame.height);

        sensor_msgs.msg.Image imageMsg = new sensor_msgs.msg.Image
        {
            Width = width,
            Height = height,
            Encoding = isColor ? "rgb8" : "mono16",
            Data = frame.GetRawTextureData(),
            Step = isColor ? width * 3 : width * 2
        };
        TimeSpan timeSinceEpoch = DateTime.Now - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        imageMsg.UpdateHeaderTime((int)timeSinceEpoch.TotalSeconds, unchecked((uint)(timeSinceEpoch.TotalMilliseconds * 1e6 % 1e9)));

        return imageMsg;
    }
}
