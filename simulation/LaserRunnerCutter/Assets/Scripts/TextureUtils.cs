using System;
using UnityEngine;
using UnityEngine.Profiling;

public class TextureUtils
{
    public static sensor_msgs.msg.Image ConvertToImageMsg(Texture2D frame)
    {
        if (frame.format != TextureFormat.RGB24 && frame.format != TextureFormat.R16)
        {
            throw new ArgumentException($"Unsupported texture format: {frame.format}");
        }

        // Unity's texture coordinates have origin at bottom left with OpenGL, so we need to
        // flip the pixels vertically
        // TODO: this has some overhead. We can just do this at the shader level.
        if (!SystemInfo.graphicsUVStartsAtTop)
        {
            FlipVertically(frame);
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

    public static void FlipVertically(Texture2D frame)
    {
        Profiler.BeginSample("FlipVertically");
        Color32[] originalPixels = frame.GetPixels32();
        // Use a temporary buffer to store flipped pixels
        Color32[] flippedPixels = new Color32[frame.width * frame.height];
        for (int y = 0; y < frame.height; y++)
        {
            int index = y * frame.width;
            int flippedIndex = (frame.height - 1 - y) * frame.width;

            // Copy the pixels to the temporary buffer
            Array.Copy(originalPixels, index, flippedPixels, flippedIndex, frame.width);
        }
        frame.SetPixels32(flippedPixels);
        Profiler.EndSample();
    }
}
