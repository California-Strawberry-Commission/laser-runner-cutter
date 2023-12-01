using UnityEngine;
using ROS2;

using camera_control_interfaces.srv;
using System;

public class ROSCamera : MonoBehaviour
{
    [SerializeField] private Camera colorCamera;
    [SerializeField] private Camera depthCamera;
    [SerializeField] private ROS2UnityComponent ros2Unity;
    [SerializeField] private int textureWidth = 848;
    [SerializeField] private int textureHeight = 480;
    [SerializeField] private int publishFps = 1;
    private RenderTexture colorRenderTexture;
    private RenderTexture depthRenderTexture;
    private ROS2Node ros2Node;
    private IService<SetExposure_Request, SetExposure_Response> setExposureSrv;
    private IPublisher<sensor_msgs.msg.Image> colorFramePub;
    private IPublisher<sensor_msgs.msg.Image> depthFramePub;

    public void SetExposure(float exposure)
    {
        colorCamera.GetComponent<ColorCamera>().SetExposure(exposure);
    }

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
                setExposureSrv = ros2Node.CreateService<SetExposure_Request, SetExposure_Response>("set_exposure", SetExposure);
                colorFramePub = ros2Node.CreatePublisher<sensor_msgs.msg.Image>("color_frame");
                depthFramePub = ros2Node.CreatePublisher<sensor_msgs.msg.Image>("depth_frame");
            }
        }

        float interval_secs = 1.0f / publishFps;
        InvokeRepeating("PublishFrame", interval_secs, interval_secs);
    }

    private SetExposure_Response SetExposure(SetExposure_Request msg)
    {
        if (msg.Exposure_ms < 0.0f)
        {
            // Negative exposure value means auto exposure. In the simulated camera
            // just assume a multiplier of 1.0
            SetExposure(1.0f);
        }
        else
        {
            // In the simulated color camera, the exposure is just a multiplier.
            // For now, just log scale 10ms to a multiplier of 1.0 and 0.001ms to 0.1
            SetExposure(0.775f + 0.0977f * (float)Math.Log(Math.E, msg.Exposure_ms));
        }
        return new SetExposure_Response();
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
