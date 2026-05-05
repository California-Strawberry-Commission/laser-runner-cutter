import dotenv from "dotenv";
import path from "node:path";

// First, load ros2 .env
dotenv.config({
  path: path.resolve(process.cwd(), "../../ros2/.env"),
});

// Then, load app-level .env.local, overriding if same keys exist
dotenv.config({
  path: path.resolve(process.cwd(), ".env.local"),
  override: true,
});

/** @type {import('next').NextConfig} */
const nextConfig = {
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "Access-Control-Allow-Credentials", value: "true" },
          { key: "Access-Control-Allow-Origin", value: "*" },
          { key: "Access-Control-Allow-Methods", value: "*" },
          { key: "Access-Control-Allow-Headers", value: "*" },
        ],
      },
    ];
  },
};

export default nextConfig;
