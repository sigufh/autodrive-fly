import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { loadEnv } from 'vite'

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    proxy: { '/api': loadEnv(mode, '.', 'AUTODRIVE_').AUTODRIVE_API_URL ?? 'http://127.0.0.1:8000' },
  },
  test: { environment: 'jsdom', setupFiles: './src/test-setup.ts', include: ['src/**/*.test.ts', 'src/**/*.test.tsx'] },
}))
