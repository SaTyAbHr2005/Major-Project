import fs from 'node:fs'
import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// HTTPS dev server only in TLS mode (VITE_API_URL is https) and only when the dev certs exist,
// so a fresh clone without certs/ still starts on plain http.
const key = new URL('../certs/server.key', import.meta.url)
const cert = new URL('../certs/server.pem', import.meta.url)

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const tls = env.VITE_API_URL?.startsWith('https:') && fs.existsSync(key) && fs.existsSync(cert)
  return {
    plugins: [react()],
    server: {
      host: 'localhost',
      port: 5173,
      https: tls ? { key: fs.readFileSync(key), cert: fs.readFileSync(cert) } : undefined,
    },
  }
})
