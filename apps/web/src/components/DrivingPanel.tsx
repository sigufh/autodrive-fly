import { useEffect, useRef } from 'react'
import type { DrivingState } from '../types'

function fmt(value: number, digits = 2) { return Number.isFinite(value) ? value.toFixed(digits) : '—' }

export function DrivingPanel({ state, running, learning, explore, safetyConstraints, scenario, onScenario, onLearning, onExplore, onSafetyConstraints, onRun, onStep, onReset }: {
  state?: DrivingState; running: boolean; learning: boolean; explore: boolean; safetyConstraints: boolean; onLearning: (value: boolean) => void; onExplore: (value: boolean) => void; onSafetyConstraints: (value: boolean) => void;
  scenario: 'highway' | 'city'; onScenario: (value: 'highway' | 'city') => void; onRun: () => void; onStep: () => void; onReset: (keep: boolean) => void;
}) {
  const road = useRef<HTMLCanvasElement>(null), retina = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    if (!state || !road.current) return
    const canvas = road.current, ctx = canvas.getContext('2d')!, { environment: env } = state
    if (env.city) {
      const [minX, minY, maxX, maxY] = env.city.map_bounds, pad = 16
      const scale = Math.min((canvas.width - pad * 2) / (maxX - minX), (canvas.height - pad * 2) / (maxY - minY))
      const px = (x: number) => pad + (x - minX) * scale, py = (y: number) => canvas.height - pad - (y - minY) * scale
      ctx.fillStyle = '#07110f'; ctx.fillRect(0, 0, canvas.width, canvas.height)
      ctx.strokeStyle = '#314a45'; ctx.lineWidth = 28; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.beginPath(); env.city.centerline.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))); ctx.stroke()
      ctx.strokeStyle = '#9bb7aa'; ctx.lineWidth = 1.5; ctx.setLineDash([5, 5]); ctx.beginPath(); env.city.centerline.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))); ctx.stroke(); ctx.setLineDash([])
      env.city.intersections.forEach(intersection => { const point = env.city!.centerline.reduce((best, value) => Math.abs(value[1] - intersection.progress) < Math.abs(best[1] - intersection.progress) ? value : best); ctx.fillStyle = intersection.signal === 'red' ? '#ff735f' : '#73f0bb'; ctx.beginPath(); ctx.arc(px(point[0]), py(point[1]), 4, 0, Math.PI * 2); ctx.fill() })
      if (env.city.world_trajectory.length > 1) { ctx.strokeStyle = '#71e4b3'; ctx.lineWidth = 2; ctx.beginPath(); env.city.world_trajectory.forEach(([x, y], i) => i ? ctx.lineTo(px(x), py(y)) : ctx.moveTo(px(x), py(y))); ctx.stroke() }
      env.city.actors.forEach(actor => { ctx.fillStyle = '#ff735f'; ctx.beginPath(); ctx.arc(px(actor.x), py(actor.y), Math.max(3, actor.radius * scale), 0, Math.PI * 2); ctx.fill() })
      ctx.fillStyle = '#ffe276'; ctx.beginPath(); ctx.arc(px(env.vehicle.world_x ?? 0), py(env.vehicle.world_y ?? 0), 5, 0, Math.PI * 2); ctx.fill()
      return
    }
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
    <header><div><span className="eyebrow">CLOSED LOOP / MALECNS</span><h2>{scenario === 'city' ? '城市路线驾驶 · Alpha' : '障碍驾驶实验'}</h2></div><span className={`mode ${running ? 'active' : ''}`}>{running ? 'RUNNING' : state?.environment.done ? 'ENDED' : 'PAUSED'}</span></header>
    <div className="sim-grid"><div><h3>{scenario === 'city' ? '城市路线 · 绿=轨迹 / 黄=车辆 / 红=路侧演员' : '神经控制轨迹 · 绿=已行驶 / 黄=短时投影'}</h3><canvas ref={road} className="road-canvas" width={320} height={500} /></div><div><h3>复眼刺激</h3><canvas ref={retina} className="retina-canvas" aria-label="48乘24视觉刺激" /><small>{state?.retina.mapped_receptors.toLocaleString() ?? '…'} R1–R6 · optic-hex proxy</small><small>PPL101 {state?.dopamine_neurons.body_ids.join(' / ') ?? '…'} · RPE gate</small></div></div>
    <div className="telemetry">
      <div><small>里程</small><strong>{fmt(state?.environment.vehicle.y ?? 0, 1)} m</strong></div>
      <div><small>速度</small><strong>{fmt(state?.environment.vehicle.speed ?? 0, 1)}</strong></div>
      <div><small>纵向指令</small><strong>{fmt(state?.action.drive ?? 0)}</strong></div>
      <div><small>MDN 后退</small><strong>{fmt(state?.action.reverse ?? 0)}</strong></div>
      <div><small>转向</small><strong>{fmt(state?.action.steering ?? 0)}</strong></div>
      <div><small>神经原始转向</small><strong>{fmt(state?.raw_action.steering ?? 0)}</strong></div>
      <div><small>安全介入</small><strong>{fmt(state?.lane_constraint.blend ?? 0)}</strong></div>
      <div><small>累计介入率</small><strong>{fmt((state?.control_statistics.constraint_rate ?? 0) * 100, 1)}%</strong></div>
      <div><small>奖励</small><strong>{fmt(state?.reward ?? 0, 3)}</strong></div>
      <div><small>已通过障碍</small><strong>{state?.environment.obstacles_passed ?? 0}/{state?.environment.obstacles.length ?? 0}</strong></div>
      <div><small>道路配对</small><strong>{state?.environment.pair_seed ?? '—'} / {state?.environment.mirror === -1 ? '镜像' : '原向'}</strong></div>
      <div><small>首障碍侧别</small><strong>{state?.environment.first_obstacle_side === 'left' ? '左' : state?.environment.first_obstacle_side === 'right' ? '右' : '—'}</strong></div>
      <div><small>首障碍通过</small><strong>{state?.environment.first_obstacle_passed ? '已通过' : '未通过'}</strong></div>
      <div><small>终止原因</small><strong>{({ obstacle: '碰撞障碍', road_boundary: '驶出道路', success: '通关', timeout: '超时' } as Record<string, string>)[state?.environment.terminal_reason ?? ''] ?? '—'}</strong></div>
      <div><small>多巴胺 RPE</small><strong className={(state?.dopamine.dopamine ?? 0) < 0 ? 'negative' : ''}>{fmt(state?.dopamine.dopamine ?? 0, 3)}</strong></div>
      <div><small>已变突触</small><strong>{state?.dopamine.changed_synapses ?? 0}/{state?.dopamine.plastic_synapses ?? 0}</strong></div>
      <div><small>双侧 PPL101</small><strong>{fmt(state?.dopamine.lateral_dopamine?.[0] ?? 0, 2)} / {fmt(state?.dopamine.lateral_dopamine?.[1] ?? 0, 2)}</strong></div>
      {state?.environment.city && <><div><small>当前道路</small><strong>{state.environment.city.road}</strong></div><div><small>下一动作</small><strong>{({ left: '左转', right: '右转', straight: '直行', arrive: '到达' } as Record<string, string>)[state.environment.city.next_maneuver] ?? state.environment.city.next_maneuver}</strong></div><div><small>交通灯 / 规则</small><strong className={state.environment.city.traffic_light === 'red' ? 'negative' : ''}>{state.environment.city.traffic_light === 'red' ? '红灯' : '绿灯'} · {state.environment.city.rule_status}</strong></div><div><small>车道任务</small><strong>{state.environment.city.rule_status === 'yield_red' ? '红灯让行' : state.environment.city.violations.length ? '违规' : '合法'}</strong></div></>}
    </div>
    <div className="driving-controls"><label>场景 <select aria-label="驾驶场景" value={scenario} disabled={running} onChange={e => onScenario(e.target.value as 'highway' | 'city')}><option value="highway">直路避障（v5 基准）</option><option value="city">城市路线 Alpha</option></select></label><label><input type="checkbox" checked={learning} onChange={e => onLearning(e.target.checked)} /> 在线可塑性（实验）</label><label><input type="checkbox" checked={explore} onChange={e => onExplore(e.target.checked)} /> 探索噪声</label><label><input type="checkbox" checked={safetyConstraints} onChange={e => onSafetyConstraints(e.target.checked)} /> 道路安全约束</label><button onClick={onRun}>{running ? '暂停' : '连续运行'}</button><button disabled={running} onClick={onStep}>单步</button><button disabled={running} onClick={() => onReset(true)}>新场景</button><button disabled={running} onClick={() => onReset(false)}>恢复发布策略</button></div>
    <p className="scientific-note">{state?.policy_checkpoint.loaded ? `已加载 ${state.policy_checkpoint.kind === 'frozen_calibrated' ? '冻结校准' : '实验学习'}策略。` : state?.policy_checkpoint.rejection ? '旧检查点已停用，当前策略未校准。' : '当前为未训练策略。'}</p>
  </section>
}
