/** @type {import('next').NextConfig} */
const nextConfig = {
  transpilePackages: [
    "@tsparticles/react",
    "@tsparticles/slim",
    "@tsparticles/engine",
  ],
  webpack: (config) => {
    config.resolve.fallback = {
      ...config.resolve.fallback,
      canvas: false,
    };
    return config;
  },
};

export default nextConfig;
