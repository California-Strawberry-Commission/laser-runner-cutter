using System;
using System.Collections.Generic;
using UnityEngine;
using ROS2;

using std_msgs.msg;
using std_srvs.srv;
using laser_control_interfaces.msg;
using laser_control_interfaces.srv;

public class ROS2Laser : MonoBehaviour
{
    private static (int lower, int upper) X_BOUNDS = (-32768, 32767);
    private static (int lower, int upper) Y_BOUNDS = (-32768, 32767);

    [SerializeField] private GameObject laserPrefab;
    [SerializeField] private float laserMaxLength = 10.0f;
    [SerializeField] private ROS2UnityComponent ros2Unity;
    private ROS2Node ros2Node;
    private IService<SetColor_Request, SetColor_Response> setColorSrv;
    private IService<GetBounds_Request, GetBounds_Response> getBoundsSrv;
    private IService<AddPoint_Request, AddPoint_Response> addPointSrv;
    private IService<SetPoints_Request, SetPoints_Response> setPointsSrv;
    private IService<Empty_Request, Empty_Response> removePointSrv;
    private IService<Empty_Request, Empty_Response> clearPointsSrv;
    private IService<SetPlaybackParams_Request, SetPlaybackParams_Response> setPlaybackParamsSrv;
    private IService<Empty_Request, Empty_Response> playSrv;
    private IService<Empty_Request, Empty_Response> stopSrv;
    private IPublisher<Bool> playingPub;
    private Color color = Color.white;
    private List<(int x, int y)> points = new List<(int, int)>();
    private List<GameObject> laserInstances = new List<GameObject>();

    public void SetColor(float r, float g, float b)
    {
        color = new Color(r, g, b);
        UpdateLaserInstancesColor();
    }

    public (int x, int y)[] GetBounds(float scale)
    {
        int xOffset = (int)Math.Round((X_BOUNDS.upper - X_BOUNDS.lower) / 2 * (1.0 - scale));
        int yOffset = (int)Math.Round((Y_BOUNDS.upper - Y_BOUNDS.lower) / 2 * (1.0 - scale));
        return new[] {
            (X_BOUNDS.lower + xOffset, Y_BOUNDS.lower + yOffset),
            (X_BOUNDS.lower + xOffset, Y_BOUNDS.upper - yOffset),
            (X_BOUNDS.upper - xOffset, Y_BOUNDS.upper - yOffset),
            (X_BOUNDS.upper - xOffset, Y_BOUNDS.lower + yOffset),
        };
    }

    public void AddPoint(int x, int y)
    {
        if (InBounds(x, y))
        {
            points.Add((x, y));
        }
    }

    public void SetPoints(List<(int, int)> points)
    {
        if (points != null)
        {
            this.points = points;
        }
    }

    public void RemovePoint()
    {
        if (points.Count > 0)
        {
            points.RemoveAt(points.Count - 1);
        }
    }

    public void ClearPoints()
    {
        points.Clear();
    }

    public void SetPlaybackParams(uint fps, uint pps, float transitionDurationMs)
    {
        // No-op
    }

    public void Play()
    {
        foreach (var point in points)
        {
            GameObject laserInstance = Instantiate(laserPrefab);
            laserInstances.Add(laserInstance);

            // TODO: convert point to ray
            Ray ray = new Ray(transform.position, transform.forward);
            RaycastHit hit;
            if (Physics.Raycast(ray, out hit, laserMaxLength))
            {
                laserInstance.transform.position = hit.point;
                laserInstance.transform.rotation = Quaternion.LookRotation(hit.normal);
            }
            else
            {
                // If the ray doesn't hit anything, extend the laser to its maximum length
                laserInstance.transform.position = ray.origin + ray.direction * laserMaxLength;
                laserInstance.transform.rotation = Quaternion.LookRotation(ray.direction);
            }
        }
        UpdateLaserInstancesColor();
        playingPub.Publish(new Bool { Data = true });
    }

    public void Stop()
    {
        foreach (GameObject laserInstance in laserInstances)
        {
            Destroy(laserInstance);
        }
        laserInstances.Clear();
        playingPub.Publish(new Bool { Data = false });
    }

    private void Start()
    {
        if (ros2Unity.Ok())
        {
            if (ros2Node == null)
            {
                ros2Node = ros2Unity.CreateNode("ROS2UnityLaserNode");
                setColorSrv = ros2Node.CreateService<SetColor_Request, SetColor_Response>("set_color", SetColor);
                getBoundsSrv = ros2Node.CreateService<GetBounds_Request, GetBounds_Response>("get_bounds", GetBounds);
                addPointSrv = ros2Node.CreateService<AddPoint_Request, AddPoint_Response>("add_point", AddPoint);
                setPointsSrv = ros2Node.CreateService<SetPoints_Request, SetPoints_Response>("set_points", SetPoints);
                removePointSrv = ros2Node.CreateService<Empty_Request, Empty_Response>("remove_point", RemovePoint);
                clearPointsSrv = ros2Node.CreateService<Empty_Request, Empty_Response>("clear_points", ClearPoints);
                setPlaybackParamsSrv = ros2Node.CreateService<SetPlaybackParams_Request, SetPlaybackParams_Response>("set_playback_params", SetPlaybackParams);
                playSrv = ros2Node.CreateService<Empty_Request, Empty_Response>("play", Play);
                stopSrv = ros2Node.CreateService<Empty_Request, Empty_Response>("stop", Stop);
                playingPub = ros2Node.CreatePublisher<Bool>("playing");
            }
        }
        playingPub.Publish(new Bool { Data = false });

        // REMOVE ME
        AddPoint(0, 0);
        AddPoint(32767, 32767);
        Play();
    }

    private SetColor_Response SetColor(SetColor_Request msg)
    {
        SetColor(msg.R, msg.G, msg.B);
        return new SetColor_Response();
    }

    private GetBounds_Response GetBounds(GetBounds_Request msg)
    {
        var bounds = GetBounds(msg.Scale);

        List<Point> points = new List<Point>();
        foreach (var point in bounds)
        {
            points.Add(new Point
            {
                X = point.x,
                Y = point.y,
            });
        }

        GetBounds_Response res = new GetBounds_Response
        {
            Points = points.ToArray()
        };
        return res;
    }

    private AddPoint_Response AddPoint(AddPoint_Request msg)
    {
        AddPoint(msg.Point.X, msg.Point.Y);
        return new AddPoint_Response();
    }

    private SetPoints_Response SetPoints(SetPoints_Request msg)
    {
        List<(int, int)> points = new List<(int, int)>();
        foreach (Point point in msg.Points)
        {
            points.Add((point.X, point.Y));
        }
        SetPoints(points);
        return new SetPoints_Response();
    }

    private Empty_Response RemovePoint(Empty_Request msg)
    {
        RemovePoint();
        return new Empty_Response();
    }

    private Empty_Response ClearPoints(Empty_Request msg)
    {
        ClearPoints();
        return new Empty_Response();
    }

    private SetPlaybackParams_Response SetPlaybackParams(SetPlaybackParams_Request msg)
    {
        SetPlaybackParams(msg.Fps, msg.Pps, msg.Transition_duration_ms);
        return new SetPlaybackParams_Response();
    }

    private Empty_Response Play(Empty_Request msg)
    {
        Play();
        return new Empty_Response();
    }

    private Empty_Response Stop(Empty_Request msg)
    {
        Stop();
        return new Empty_Response();
    }

    private bool InBounds(int x, int y)
    {
        return x >= X_BOUNDS.lower && x <= X_BOUNDS.upper && y >= Y_BOUNDS.lower && y <= Y_BOUNDS.upper;
    }

    private void UpdateLaserInstancesColor()
    {
        foreach (GameObject laserInstance in laserInstances)
        {
            Material material = laserInstance.GetComponent<Renderer>().material;
            material.SetColor("_Color", color);
            material.SetColor("_EmissionColor", color);
        }
    }
}
