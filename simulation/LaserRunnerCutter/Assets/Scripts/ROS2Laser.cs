using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ROS2Laser : MonoBehaviour
{
    public GameObject laserPrefab;
    public float laserMaxLength = 10.0f;

    private GameObject laserInstance;

    private void Start()
    {
        Play();
    }

    private void Play()
    {
        if (laserInstance == null)
        {
            laserInstance = Instantiate(laserPrefab);
        }

        // X and Y bounds: [0, 4095] (12 bits)
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

    private void Stop()
    {
        if (laserInstance != null)
        {
            Destroy(laserInstance);
            laserInstance = null;
        }
    }
}
