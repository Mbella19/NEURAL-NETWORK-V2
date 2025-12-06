/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    optimizePackageImports: ['recharts', 'framer-motion'],
  },
}

module.exports = nextConfig
