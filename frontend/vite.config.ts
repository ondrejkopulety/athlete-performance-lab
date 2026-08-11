import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Ve vývoji jede API na 8000 a frontend na 5173; proxy je tady proto, aby
// se v kódu volalo `/api/...` stejně jako v produkci za nginxem – žádné
// přepínání base URL a žádné CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
