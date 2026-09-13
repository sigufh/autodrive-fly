import { afterEach, expect, test, vi } from 'vitest'
import { streamDriving } from './api'
import type { DrivingState } from './types'
import { stateColor, topologyCurve } from './components/CnsViewer'

afterEach(() => vi.unstubAllGlobals())
const state = { environment: { step: 1 } } as DrivingState
test('handles split UTF-8 NDJSON driving chunks and completion', async () => {
  const payload = new TextEncoder().encode(`${JSON.stringify({ type: 'driving_state', ...state })}\n{"type":"done"}`)
  const events: DrivingState[] = []
  vi.stubGlobal('fetch', vi.fn(async () => new Response(new ReadableStream({ start(controller) {
    for (let i = 0; i < payload.length; i += 2) controller.enqueue(payload.slice(i, i + 2))
    controller.close()
  } }))))
  await streamDriving(true, true, true, 'assisted', event => events.push(event))
  expect(events[0].environment.step).toBe(1)
})
test('rejects a truncated driving stream', async () => {
  vi.stubGlobal('fetch', vi.fn(async () => new Response(`${JSON.stringify({ type: 'driving_state', ...state })}\n`)))
  await expect(streamDriving(true, true, true, 'assisted', () => {})).rejects.toThrow('驾驶数据流中断')
})
test('execution mode sends no learning or exploration by default contract', async () => {
  let body = ''
  vi.stubGlobal('fetch', vi.fn(async (_url, init) => {
    body = String(init?.body)
    return new Response('{"type":"done"}')
  }))
  await streamDriving(false, false, true, 'neural', () => {})
  expect(JSON.parse(body)).toMatchObject({ learning: false, explore: false, safety_constraints: true, control_mode: 'neural' })
})
test('colors reflect sign and magnitude', () => {
  expect(stateColor(1, 1)).not.toEqual(stateColor(-1, 1))
  expect(stateColor(1, 1).reduce((a, b) => a + b)).toBeGreaterThan(stateColor(0.001, 1).reduce((a, b) => a + b))
})
test('topology self-loop has nonzero extent', () => {
  expect(topologyCurve({ x: 100, y: 100 }, { x: 100, y: 100 }, true)).toContain('C')
})
