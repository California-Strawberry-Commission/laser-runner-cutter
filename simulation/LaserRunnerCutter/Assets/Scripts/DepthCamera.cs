using UnityEngine;

public class DepthCamera : MonoBehaviour
{
    [SerializeField] Shader shader;
    private Material material;

    private void Start()
    {
        Camera camera = GetComponent<Camera>();
        camera.depthTextureMode = DepthTextureMode.Depth;

        if (shader == null)
        {
            shader = Shader.Find("Hidden/DepthCamera");
        }
        material = new Material(shader);
    }

    private void OnRenderImage(RenderTexture source, RenderTexture dest)
    {
        if (material != null)
        {
            Graphics.Blit(source, dest, material);
        }
    }
}
