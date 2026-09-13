import type { CnsOverview, DrivingEvent, DrivingState, PathwayOverview, SkeletonResponse } from './types'

async function checked<T>(response: Response): Promise<T> {
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`)
  return response.json() as Promise<T>
}
export function fetchSkeleton(bodyId: number) { return fetch(`/api/skeleton/${bodyId}?max_edges=20000`).then(checked<SkeletonResponse>) }
export function fetchOverview() { return fetch('/api/connectome/overview').then(checked<CnsOverview>) }
export function fetchPathways() { return fetch('/api/connectome/pathways').then(checked<PathwayOverview>) }
export function fetchDrivingState() { return fetch('/api/driving/state').then(checked<DrivingState>) }
export function resetDriving(seed: number, keepLearning: boolean) {
  return fetch('/api/driving/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ seed, keep_learning: keepLearning }) }).then(checked<DrivingState>)
}
export function stepDriving(steps: number, learning: boolean, explore: boolean, safetyConstraints: boolean) {
  return fetch('/api/driving/step', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ steps, learning, explore, safety_constraints: safetyConstraints }) }).then(checked<DrivingState>)
}
export async function streamDriving(learning: boolean, explore: boolean, safetyConstraints: boolean, onState: (state: DrivingState) => void, signal?: AbortSignal) {
  const response = await fetch('/api/driving/run-stream', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ max_steps: 500, learning, explore, safety_constraints: safetyConstraints }), signal,
  })
  if (!response.ok || !response.body) throw new Error(`${response.status}: ${await response.text()}`)
  const reader = response.body.getReader(), decoder = new TextDecoder()
  let buffer = '', complete = false
  const consume = (line: string) => {
    if (!line.trim()) return
    const event = JSON.parse(line) as DrivingEvent
    if (event.type === 'done') complete = true
    else if (event.type === 'error') throw new Error(event.message)
    else onState(event)
  }
  try {
    for (;;) {
      const { value, done } = await reader.read()
      buffer += decoder.decode(value, { stream: !done })
      const lines = buffer.split('\n'); buffer = lines.pop() ?? ''; lines.forEach(consume)
      if (done) break
    }
    consume(buffer)
    if (!complete) throw new Error('驾驶数据流中断。')
  } finally { await reader.cancel(); reader.releaseLock() }
}
