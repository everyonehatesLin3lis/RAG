import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Static export: `npm run build` writes plain HTML, JS and CSS to out/, which Firebase Hosting serves as files.
  // Possible because the page needs no server of its own: it runs in the browser and talks to the FastAPI backend.
  // NEXT_PUBLIC_API_BASE_URL is read at build time, so set it before building for the deployment.
  output: "export",
};

export default nextConfig;
