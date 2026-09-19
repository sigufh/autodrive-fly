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
  await expect(page.getByLabel('实时因果链')).toContainText('当前运行链 · assisted-v5')
  await expect(page.getByLabel('实时因果链')).toContainText('工程辅助')
  const v7 = page.getByLabel('v7 离线验证状态')
  await expect(v7).toContainText('当前阶段：controlled_vision')
  await expect(v7).toContainText('MaleCNS 柱映射：Tm9 532266 → [15, 2]；CT1 Lo1 逐突触列可用，全官方 LO 覆盖不完整')
  await expect(v7).toContainText('未部署')
  await expect(v7).toContainText('T4_T5_direction_and_ON_OFF · STOP')
  await expect(v7).toContainText('九源合同：INCOMPLETE')
  await expect(v7).toContainText('Mi4/C3 直接数值电压：Groschner_2022；独立候选 0')
  await expect(v7).toContainText('CT1 电压候选：审计 11；2025–2026 新候选 3；直接实验电压未命中')
  await expect(page.getByText('PuRe ZIP：已核验；新增数值载荷未命中')).toBeVisible()
  await expect(v7).toContainText('T5 字段：flash 7/15 · white-noise 8/15 · grating 9/15')
  await expect(page.getByText('Braun 钙载荷：Tm2 9 flies · Tm9 11 · CT1 9；条件网格完整；合规电压源 0')).toBeVisible()
  await expect(v7).toContainText('Motyxia2 历史：22 branches / 447 commits；逐记录日志未命中')
  await expect(v7).toContainText('publisher supplements 已检查 / record log 未命中')
  await expect(v7).toContainText('PMC endpoint 不可访问')
  await expect(v7).toContainText('全局不存在未声明')
  await expect(v7).toContainText('生成器默认值未代填')
  await expect(v7).toContainText('EPG_PEN_PEG_heading · PASS')
  await expect(v7).toContainText('果蝇局部核：组件证据，未获发布授权')
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
  await expect(page.getByText('R1–R6 · optic-hex proxy', { exact: false })).toBeVisible()
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
  await expect(page.getByText('已加载发布态 MaleCNS v6，默认服务与 v5/v6 检查点未改变。v7 仅离线实验：R1–R6 神经闭环、EPG/PEN/PEG 航向环和 FC2/PFL3/DNa02 透明链已有组件级实测与因果对照，但新布局的留一镜像对验证失败；严格 T4/T5 方向与 ON/OFF、LPLC1/LPLC2/LC4 分型也未通过。v7 未部署、未进入 MB 学习或外部 final。')).toBeVisible({ timeout: 120000 })
  await expect(page.getByText('障碍驾驶实验')).toBeVisible()
  await expect(page.locator('.road-canvas')).toBeVisible()
  await expect(page.getByText('MaleCNS v6', { exact: true })).toBeVisible()
  await expect(page.getByText('8% / step', { exact: true })).toBeVisible()
  await expect(page.getByText('前视 143°', { exact: true })).toBeVisible()
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
