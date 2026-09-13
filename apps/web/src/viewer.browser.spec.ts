import { test, expect } from '@playwright/test'
import { loadEnv } from 'vite'

const env = loadEnv('test', '.', 'AUTODRIVE_')
const webUrl = env.AUTODRIVE_WEB_URL ?? 'http://127.0.0.1:5174'

test.beforeEach(async ({ request }) => {
  const response = await request.post(`${webUrl}/api/driving/reset`, {
    data: { seed: 400, keep_learning: false },
  })
  expect(response.ok()).toBe(true)
  const state = await response.json()
  expect(state.environment.pair_seed).toBe(200)
  expect(state.environment.first_obstacle_side).toBe('left')
  expect(state.policy_checkpoint).toHaveProperty('rejection')
  expect(state.policy_checkpoint.loaded).toBe(true)
  expect(state.policy_checkpoint.kind).toBe('learned_v5')
})

test('real topology coverage, driving activity and stable scene', async ({ page }, testInfo) => {
  const errors: string[] = []
  page.on('pageerror', e => errors.push(e.message))
  await page.goto(webUrl)
  await expect(page.locator('.canvas-host canvas')).toBeVisible({ timeout: 120000 })
  await expect(page.getByText('BODY 10059', { exact: true })).toBeVisible()
  const canvas = page.locator('.canvas-host canvas')
  await canvas.evaluate(el => el.setAttribute('data-scene-instance', 'original'))
  await page.getByRole('button', { name: '完整拓扑图' }).click()
  const svg = page.getByLabel('全通路拓扑图')
  await expect(svg).toHaveAttribute('data-edge-coverage', '25582938')
  expect(await svg.locator('path[data-source]').count()).toBeGreaterThan(198)
  expect(await svg.locator('path[data-source]').evaluateAll(nodes => nodes.filter(n => n.getAttribute('data-source') === n.getAttribute('data-target')).length)).toBeGreaterThan(0)
  await page.screenshot({ path: testInfo.outputPath('review-topology.png'), fullPage: true })
  await page.getByRole('button', { name: '解剖 3D', exact: true }).click()
  await page.getByRole('button', { name: '单步' }).click()
  await expect(canvas).toHaveAttribute('data-displayed-nodes', '220', { timeout: 120000 })
  await expect(page.getByText('R1–R6', { exact: false })).toBeVisible()
  const cameraPosition = await canvas.getAttribute('data-camera-position')
  await page.getByLabel('神经元 ID').fill('10002')
  await page.getByRole('button', { name: '加载骨架' }).click()
  await expect(page.getByText('BODY 10002', { exact: true })).toBeVisible({ timeout: 120000 })
  await expect(canvas).toHaveAttribute('data-scene-instance', 'original')
  await expect(canvas).toHaveAttribute('data-camera-position', cameraPosition!)
  await expect(canvas).toHaveAttribute('data-displayed-nodes', '220')
  await expect(page.getByRole('button', { name: '40K 强连接' })).toHaveAttribute('aria-pressed', 'false')
  await page.screenshot({ path: testInfo.outputPath('review-anatomy.png'), fullPage: true })
  expect(errors).toEqual([])
})

test('driving controls expose learning, stepping and reset', async ({ page }) => {
  await page.goto(webUrl)
  await expect(page.getByText('障碍驾驶实验')).toBeVisible({ timeout: 120000 })
  await expect(page.getByLabel('在线可塑性（实验）')).not.toBeChecked()
  await expect(page.getByLabel('探索噪声')).not.toBeChecked()
  await expect(page.getByLabel('道路安全约束')).toBeChecked()
  await page.getByRole('button', { name: '单步' }).click()
  await expect(page.locator('.telemetry').getByText(/\/4383$/)).toBeVisible({ timeout: 120000 })
  await page.getByRole('button', { name: '恢复发布策略' }).click()
  await expect(page.locator('.telemetry').getByText(/[1-9]\d*\/4383$/)).toBeVisible({ timeout: 120000 })
})

test('neural decision experiment keeps the random-obstacle map', async ({ page }) => {
  await page.goto(webUrl)
  await page.getByLabel('控制模式').selectOption('neural')
  await expect(page.getByText('已加载 MaleCNS v6 单障碍课程策略：运动神经输出直接映射车辆；三障碍已零样本通过，完整九障碍尚未完成。')).toBeVisible({ timeout: 120000 })
  await expect(page.getByText('障碍驾驶实验')).toBeVisible()
  await expect(page.locator('.road-canvas')).toBeVisible()
  await expect(page.getByText('MaleCNS v6', { exact: true })).toBeVisible()
})

test('mirror telemetry stays visible on mobile with nonblank road and retina', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto(webUrl)
  await expect(page.getByText('首障碍侧别', { exact: true })).toBeVisible({ timeout: 120000 })
  await page.getByRole('button', { name: '单步', exact: true }).click()
  await expect(page.locator('.telemetry').getByText(/[1-9]\d*\/4383$/)).toBeVisible({ timeout: 120000 })
  for (const selector of ['.road-canvas', '.retina-canvas']) {
    const colors = await page.locator(selector).evaluate((node: HTMLCanvasElement) => {
      const pixels = node.getContext('2d')!.getImageData(0, 0, node.width, node.height).data
      const unique = new Set<number>()
      for (let i = 0; i < pixels.length; i += 4) unique.add((pixels[i] << 16) | (pixels[i + 1] << 8) | pixels[i + 2])
      return unique.size
    })
    expect(colors).toBeGreaterThan(3)
  }
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
  await page.screenshot({ path: testInfo.outputPath('mirror-mobile.png'), fullPage: true })
})
