Shader "Hidden/DepthCamera" {
  SubShader {
    Tags { "RenderType"="Opaque" }
    Pass {
      CGPROGRAM

      #pragma vertex vert
      #pragma fragment frag
      #include "UnityCG.cginc"

      struct appdata {
        float4 vertex : POSITION;
        float2 uv : TEXCOORD0;
      };

      struct v2f {
        float4 vertex : SV_POSITION;
        float2 uv : TEXCOORD0;
      };

      sampler2D _CameraDepthTexture;

      v2f vert (appdata v) {
        v2f o;
        UNITY_INITIALIZE_OUTPUT(v2f, o);
        o.vertex = UnityObjectToClipPos(v.vertex);
        o.uv = v.uv;
        return o;
      }

      half4 frag (v2f i) : SV_Target {
        float2 uv = i.uv;
        #if !UNITY_UV_STARTS_AT_TOP
        uv.y = 1.0 - uv.y;
        #endif
        // Get depth from depth texture
        float depth = UNITY_SAMPLE_DEPTH(tex2D(_CameraDepthTexture, uv));
        depth = Linear01Depth(depth);
        return depth;
      }
      ENDCG
    }
  }
}
