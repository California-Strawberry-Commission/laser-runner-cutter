using UnityEngine;
using ROS2;

public class ROSCamera : MonoBehaviour
{
    public Camera colorCamera;
    public Camera depthCamera;
    public ROS2UnityComponent ros2Unity;
    public int textureWidth = 848;
    public int textureHeight = 480;
    public int publishFps = 1;

    private RenderTexture colorRenderTexture;
    private RenderTexture depthRenderTexture;
    private ROS2Node ros2Node;
    private IPublisher<sensor_msgs.msg.Image> colorFramePub;
    private IPublisher<sensor_msgs.msg.Image> depthFramePub;

    private void Start()
    {
        colorRenderTexture = new RenderTexture(textureWidth, textureHeight, 16);
        colorCamera.targetTexture = colorRenderTexture;
        colorCamera.depthTextureMode = DepthTextureMode.None;

        depthRenderTexture = new RenderTexture(textureWidth, textureHeight, 16);
        depthCamera.targetTexture = depthRenderTexture;
        depthCamera.depthTextureMode = DepthTextureMode.Depth;

        if (ros2Unity.Ok())
        {
            if (ros2Node == null)
            {
                ros2Node = ros2Unity.CreateNode("ROS2UnityCameraNode");
                colorFramePub = ros2Node.CreatePublisher<sensor_msgs.msg.Image>("color_frame");
                depthFramePub = ros2Node.CreatePublisher<sensor_msgs.msg.Image>("depth_frame");
            }
        }

        float interval_secs = 1.0f / publishFps;
        InvokeRepeating("PublishFrame", interval_secs, interval_secs);
    }

    private void PublishFrame()
    {
        sensor_msgs.msg.Image colorImageMsg = TextureUtils.ConvertToImageMsg(GetColorFrame());
        sensor_msgs.msg.Image depthImageMsg = TextureUtils.ConvertToImageMsg(GetDepthFrame());
        colorFramePub.Publish(colorImageMsg);
        depthFramePub.Publish(depthImageMsg);
    }

    private Texture2D GetColorFrame()
    {
        Texture2D frame = new Texture2D(colorRenderTexture.width, colorRenderTexture.height, TextureFormat.RGB24, false);

        // Texture2D.ReadPixels looks at the active RenderTexture.
        RenderTexture oldActiveRenderTexture = RenderTexture.active;
        RenderTexture.active = colorRenderTexture;

        frame.ReadPixels(new Rect(0, 0, colorRenderTexture.width, colorRenderTexture.height), 0, 0);

        // Restore active RT
        RenderTexture.active = oldActiveRenderTexture;

        return frame;
    }

    private Texture2D GetDepthFrame()
    {
        Texture2D frame = new Texture2D(depthRenderTexture.width, depthRenderTexture.height, TextureFormat.R16, false);

        // Texture2D.ReadPixels looks at the active RenderTexture.
        RenderTexture oldActiveRenderTexture = RenderTexture.active;
        RenderTexture.active = depthRenderTexture;

        frame.ReadPixels(new Rect(0, 0, depthRenderTexture.width, depthRenderTexture.height), 0, 0);

        // Restore active RT
        RenderTexture.active = oldActiveRenderTexture;

        return frame;
    }
}
