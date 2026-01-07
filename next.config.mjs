/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    "@tsparticles/react",
    "@tsparticles/slim",
    "tsparticles-engine"
  ],
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      canvas: false, // Ignores node-only 'canvas' module during build
    };
    return config;
  },
};

export default nextConfig;
