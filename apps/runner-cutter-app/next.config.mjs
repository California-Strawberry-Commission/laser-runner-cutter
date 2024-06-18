/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
      return [
          {
              // matching all API routes
              source: "/:path*",
              headers: [
                  { key: "Access-Control-Allow-Credentials", value: "true" },
                  { key: "Access-Control-Allow-Origin", value: "*" },
                  { key: "Access-Control-Allow-Methods", value: "*" },
                  { key: "Access-Control-Allow-Headers", value: "*" },
              ]
          }
      ]
  }
}

export default nextConfig;
