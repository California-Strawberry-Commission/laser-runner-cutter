using System;
using UnityEngine;
using ROS2;

// using sensor_msgs.msg.Image;

public class ROSCamera : MonoBehaviour
{
    public Camera mainCamera;
    public ROS2UnityComponent ros2Unity;
    public int textureWidth = 848;
    public int textureHeight = 480;
    public int publishFps = 1;

    private RenderTexture _renderTexture;
    private ROS2Node _ros2Node;
    private IPublisher<sensor_msgs.msg.Image> _colorFramePub;

    private void Start()
    {
        if (mainCamera == null)
        {
            mainCamera = GetComponent<Camera>();
        }

        _renderTexture = new RenderTexture(textureWidth, textureHeight, 16);
        mainCamera.targetTexture = _renderTexture;

        if (ros2Unity.Ok())
        {
            if (_ros2Node == null)
            {
                _ros2Node = ros2Unity.CreateNode("ROS2UnityCameraNode");
                _colorFramePub = _ros2Node.CreatePublisher<sensor_msgs.msg.Image>("color_frame");
            }
        }

        float interval_secs = 1.0f / publishFps;
        InvokeRepeating("PublishFrame", interval_secs, interval_secs);
    }

    private void PublishFrame()
    {
        sensor_msgs.msg.Image msg = ConvertToImageMsg(GetFrame());
        _colorFramePub.Publish(msg);
    }

    private Texture2D GetFrame()
    {
        // Texture2D.ReadPixels looks at the active RenderTexture.
        RenderTexture oldActiveRenderTexture = RenderTexture.active;
        RenderTexture.active = _renderTexture;

        Texture2D frame = new Texture2D(_renderTexture.width, _renderTexture.height, TextureFormat.RGB24, false);
        frame.ReadPixels(new Rect(0, 0, _renderTexture.width, _renderTexture.height), 0, 0);

        // Restore active RT
        RenderTexture.active = oldActiveRenderTexture;

        return frame;
    }

    private sensor_msgs.msg.Image ConvertToImageMsg(Texture2D frame)
    {
        // Unity's texture coordinates have origin at bottom left, so we need to
        // flip the pixels vertically
        Color[] pixels = frame.GetPixels();
        for (int i = 0; i < frame.width; i++)
        {
            for (int j = 0; j < frame.height / 2; j++)
            {
                int index1 = j * frame.width + i;
                int index2 = (frame.height - 1 - j) * frame.width + i;

                // Swap the pixels
                Color temp = pixels[index1];
                pixels[index1] = pixels[index2];
                pixels[index2] = temp;
            }
        }
        frame.SetPixels(pixels);

        sensor_msgs.msg.Image imageMsg = new sensor_msgs.msg.Image
        {
            Width = unchecked((uint)frame.width),
            Height = unchecked((uint)frame.height),
            Encoding = "rgb8",
            Data = frame.GetRawTextureData(),
            Step = unchecked((uint)frame.width) * 3,
        };
        TimeSpan timeSinceEpoch = DateTime.Now - new DateTime(1970, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        imageMsg.UpdateHeaderTime((int)timeSinceEpoch.TotalSeconds, unchecked((uint)(timeSinceEpoch.TotalMilliseconds * 1e6 % 1e9)));

        return imageMsg;
    }
}
