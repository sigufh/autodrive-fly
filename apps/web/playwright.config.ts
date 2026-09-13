import { defineConfig } from '@playwright/test'

export default defineConfig({
  testDir: './src',
  testMatch: '**/*.browser.spec.ts',
  timeout: 180000,
  use: {
    viewport: { width: 1440, height: 1000 },
    headless: true,
    launchOptions: { executablePath: '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge' },
  },
})
