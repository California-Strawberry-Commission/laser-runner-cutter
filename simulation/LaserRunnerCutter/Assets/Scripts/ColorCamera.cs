using UnityEngine;

public class ColorCamera : MonoBehaviour
{
    [SerializeField] private Shader shader;
    [SerializeField] private float exposure = 1.0f;
    private Material material;

    public void SetExposure(float exposure)
    {
        this.exposure = exposure;
        if (material != null)
        {
            material.SetFloat("_Exposure", exposure);
        }
    }

    private void Start()
    {
        if (shader == null)
        {
            shader = Shader.Find("Hidden/ColorCamera");
        }
        material = new Material(shader);
        material.SetFloat("_Exposure", exposure);
    }

    private void OnRenderImage(RenderTexture source, RenderTexture dest)
    {
        if (material != null)
        {
            Graphics.Blit(source, dest, material);
        }
    }
}
