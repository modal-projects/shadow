import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  root: ".",
  build: { outDir: "dist", emptyOutDir: true, target: "es2022" },
  server: {
    host: "127.0.0.1",
    port: 5317,
    strictPort: true,
    proxy: {
      "/api": {
        target: process.env.SHADOW_DEMO_BACKEND || "https://shadow.modal.dev",
        changeOrigin: true,
      },
      "/replays": {
        target: process.env.SHADOW_DEMO_BACKEND || "https://shadow.modal.dev",
        changeOrigin: true,
      },
      "/rig.json": {
        target: process.env.SHADOW_DEMO_BACKEND || "https://shadow.modal.dev",
        changeOrigin: true,
      },
    },
  },
});
