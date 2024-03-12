Shader "Hidden/ColorCamera" {
  Properties {
    _MainTex("Texture", 2D) = "white" {}
    _Exposure("Exposure", Range(0.0, 1.0)) = 1.0
  }
  SubShader {
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

      sampler2D _MainTex;
      uniform float _Exposure;

      v2f vert (appdata v) {
        v2f o;
        UNITY_INITIALIZE_OUTPUT(v2f, o);
        o.vertex = UnityObjectToClipPos(v.vertex);
        o.uv = v.uv;
        return o;
      }

      fixed4 frag (v2f i) : SV_Target {
        float2 uv = i.uv;
        #if !UNITY_UV_STARTS_AT_TOP
        uv.y = 1.0 - uv.y;
        #endif
        fixed4 color = tex2D(_MainTex, uv);
        // Multiply rgb by exposure
        fixed3 rgb = color.rgb * _Exposure;
        color = fixed4(rgb, color.a);
        return color;
      }
      ENDCG
    }
  }
}
