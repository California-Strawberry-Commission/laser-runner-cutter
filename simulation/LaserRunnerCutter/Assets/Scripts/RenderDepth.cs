using UnityEngine;

public class RenderDepth : MonoBehaviour
{
    private Shader shader;
    private Material material;

    private void Start()
    {
        Camera camera = GetComponent<Camera>();
        camera.depthTextureMode = DepthTextureMode.Depth;

        shader = Shader.Find("Hidden/Depth");
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
