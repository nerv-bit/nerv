/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    "@tsparticles/react",
    "@tsparticles/slim",
    "@tsparticles/engine",
    "tsparticles-engine"
  ],
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      canvas: false,
      fs: false, // Optional: add if you see fs-related errors
    };
    return config;
  },
};

export default nextConfig;
