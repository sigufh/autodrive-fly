import { useEffect, useRef } from 'react'
import type { DrivingState } from '../types'

function fmt(value: number, digits = 2) { return Number.isFinite(value) ? value.toFixed(digits) : '—' }

export function DrivingPanel({ state, running, learning, onLearning, onRun, onStep, onReset }: {
  state?: DrivingState; running: boolean; learning: boolean; onLearning: (value: boolean) => void;
  onRun: () => void; onStep: () => void; onReset: (keep: boolean) => void;
}) {
  const road = useRef<HTMLCanvasElement>(null), retina = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    if (!state || !road.current) return
    const canvas = road.current, ctx = canvas.getContext('2d')!, { environment: env } = state
    const sx = canvas.width / (env.road_half_width * 2 + 5), sy = canvas.height / env.road_length
    const px = (x: number) => canvas.width / 2 + x * sx, py = (y: number) => canvas.height - y * sy
    ctx.fillStyle = '#07110f'; ctx.fillRect(0, 0, canvas.width, canvas.height)
    ctx.fillStyle = '#172c29'; ctx.fillRect(px(-env.road_half_width), 0, env.road_half_width * 2 * sx, canvas.height)
    ctx.strokeStyle = '#8eb5aa'; ctx.lineWidth = 2; ctx.setLineDash([8, 10]); ctx.beginPath(); ctx.moveTo(px(0), 0); ctx.lineTo(px(0), canvas.height); ctx.stroke(); ctx.setLineDash([])
    if (env.trajectory.length > 1) { ctx.strokeStyle = '#71e4b3'; ctx.lineWidth = 2; ctx.beginPath(); env.trajectory.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))); ctx.stroke() }
    if (env.projected_trajectory.length) { ctx.strokeStyle = '#ffe276'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(px(env.vehicle.x), py(env.vehicle.y)); env.projected_trajectory.forEach(([x, y]) => ctx.lineTo(px(x), py(y))); ctx.stroke(); ctx.setLineDash([]) }
    ctx.fillStyle = '#ff735f'; for (const o of env.obstacles) { ctx.beginPath(); ctx.arc(px(o.x), py(o.y), Math.max(3, o.radius * sx), 0, Math.PI * 2); ctx.fill() }
    ctx.save(); ctx.translate(px(env.vehicle.x), py(env.vehicle.y)); ctx.rotate(env.vehicle.heading); ctx.fillStyle = '#73f0bb'; ctx.beginPath(); ctx.moveTo(0, -9); ctx.lineTo(-6, 8); ctx.lineTo(6, 8); ctx.closePath(); ctx.fill(); ctx.restore()
  }, [state])
  useEffect(() => {
    if (!state || !retina.current) return
    const image = state.retina.stimulus, canvas = retina.current, ctx = canvas.getContext('2d')!
    canvas.width = state.retina.width; canvas.height = state.retina.height
    const data = ctx.createImageData(canvas.width, canvas.height)
    image.flat().forEach((value, index) => { const v = Math.round(Math.min(1, Math.max(0, value)) * 255); data.data[index * 4] = v; data.data[index * 4 + 1] = v; data.data[index * 4 + 2] = v; data.data[index * 4 + 3] = 255 })
    ctx.putImageData(data, 0, 0)
  }, [state])
  return <section className="driving-panel" aria-label="果蝇视觉驾驶">
    <header><div><span className="eyebrow">CLOSED LOOP / MALECNS</span><h2>障碍驾驶实验</h2></div><span className={`mode ${running ? 'active' : ''}`}>{running ? 'RUNNING' : state?.environment.done ? 'ENDED' : 'PAUSED'}</span></header>
    <div className="sim-grid"><div><h3>神经控制轨迹 · 绿=已行驶 / 黄=短时投影</h3><canvas ref={road} className="road-canvas" width={320} height={500} /></div><div><h3>复眼刺激</h3><canvas ref={retina} className="retina-canvas" aria-label="48乘24视觉刺激" /><small>{state?.retina.mapped_receptors.toLocaleString() ?? '…'} R1–R6 · optic-hex proxy</small><small>PPL101 {state?.dopamine_neurons.body_ids.join(' / ') ?? '…'} · RPE gate</small></div></div>
    <div className="telemetry">
      <div><small>里程</small><strong>{fmt(state?.environment.vehicle.y ?? 0, 1)} m</strong></div>
      <div><small>速度</small><strong>{fmt(state?.environment.vehicle.speed ?? 0, 1)}</strong></div>
      <div><small>转向</small><strong>{fmt(state?.action.steering ?? 0)}</strong></div>
      <div><small>奖励</small><strong>{fmt(state?.reward ?? 0, 3)}</strong></div>
      <div><small>多巴胺 RPE</small><strong className={(state?.dopamine.dopamine ?? 0) < 0 ? 'negative' : ''}>{fmt(state?.dopamine.dopamine ?? 0, 3)}</strong></div>
      <div><small>已变突触</small><strong>{state?.dopamine.changed_synapses ?? 0}/{state?.dopamine.plastic_synapses ?? 0}</strong></div>
      <div><small>双侧 PPL101</small><strong>{fmt(state?.dopamine.lateral_dopamine?.[0] ?? 0, 2)} / {fmt(state?.dopamine.lateral_dopamine?.[1] ?? 0, 2)}</strong></div>
    </div>
    <div className="driving-controls"><label><input type="checkbox" checked={learning} onChange={e => onLearning(e.target.checked)} /> 多巴胺可塑性</label><button onClick={onRun}>{running ? '暂停' : '连续运行'}</button><button disabled={running} onClick={onStep}>单步</button><button disabled={running} onClick={() => onReset(true)}>新场景</button><button disabled={running} onClick={() => onReset(false)}>清除学习</button></div>
    <p className="scientific-note">{state?.policy_checkpoint.loaded ? '已加载通过正收益门槛的策略检查点。' : '当前为未训练策略。'} 每个车辆动作前运行 {state?.motor.brain_substeps_per_action ?? 4} 个 CNS 微步。活动是持续的无量纲模型状态；双侧 PPL101 是训练调制信号。DNp20 / DNpe017 的车辆控制含义是工程读出，不是已验证的天然驾驶功能。</p>
  </section>
}
