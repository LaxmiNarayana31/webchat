import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const backendTarget =
    env.VITE_BACKEND_TARGET || env.VITE_API_TARGET || 'http://localhost:8000'

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
