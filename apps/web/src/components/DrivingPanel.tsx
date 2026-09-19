import { useEffect, useRef } from 'react'
import type { DrivingState, V7Status } from '../types'

function fmt(value: number, digits = 2) { return Number.isFinite(value) ? value.toFixed(digits) : '—' }

export function DrivingPanel({ state, v7Status, v7StatusError, running, learning, explore, safetyConstraints, controlMode, onControlMode, onLearning, onExplore, onSafetyConstraints, onRun, onStep, onReset }: {
  state?: DrivingState; running: boolean; learning: boolean; explore: boolean; safetyConstraints: boolean; onLearning: (value: boolean) => void; onExplore: (value: boolean) => void; onSafetyConstraints: (value: boolean) => void;
  v7Status?: V7Status;
  v7StatusError: boolean;
  controlMode: 'assisted' | 'neural'; onControlMode: (value: 'assisted' | 'neural') => void; onRun: () => void; onStep: () => void; onReset: (keep: boolean) => void;
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
    <section className="causal-chain" aria-label="实时因果链"><h3>当前运行链 · {controlMode === 'neural' ? 'neural-v6-front' : 'assisted-v5'}</h3><div><span>视觉输入</span><strong>R1–R6 {state?.retina.mapped_receptors.toLocaleString() ?? '…'}</strong><i>→</i><span>神经响应</span><strong>T4/T5 flow {fmt(state?.motor.sensory_projection?.T4_T5_horizontal_flow ?? 0, 3)} · |x|max {state?.activity?.statistics.max_abs_state.toExponential(2) ?? '…'}</strong><i>→</i><span>神经意图</span><strong>{fmt(state?.raw_action.steering ?? 0)}</strong>{controlMode === 'assisted' && <><i>＋</i><span>工程辅助</span><strong>障碍 {fmt(state?.lane_constraint.visual_avoidance ?? 0)} · 道路 {fmt(state?.lane_constraint.road_recovery ?? 0)}</strong></>}<i>→</i><span>固定映射</span><strong>{state?.motor.mapping ?? '…'}</strong><i>→</i><span>安全约束差值</span><strong>{fmt((state?.action.steering ?? 0) - (state?.raw_action.steering ?? 0))}</strong><i>→</i><span>执行动作</span><strong>{fmt(state?.action.steering ?? 0)}</strong></div></section>
    <section className="v7-status-card" aria-label="v7 离线验证状态"><header><div><span className="eyebrow">V7 OFFLINE EVIDENCE</span><h3>模块门状态</h3></div><strong>{v7Status ? v7Status.deployment_enabled ? '已部署' : '未部署' : v7StatusError ? '证据不可用' : '读取中'}</strong></header><p>{v7StatusError ? '哈希验证证据不可用；不显示推断状态。' : `当前阶段：${v7Status?.current_stage ?? '读取审计…'} · 默认服务${v7Status?.default_runtime_changed ? '已改变' : '未改变'}`}</p><div className="gate-grid">{v7Status ? Object.entries(v7Status.gates).map(([name, passed]) => <span key={name} className={passed ? 'gate-pass' : 'gate-fail'}>{name} · {passed ? 'PASS' : 'STOP'}</span>) : !v7StatusError && <span>读取哈希验证证据…</span>}</div>{v7Status && <div className="evidence-boundaries" aria-label="v7 证据边界"><strong>九源合同：{v7Status.evidence_boundaries.nine_source_contract_complete ? 'COMPLETE' : 'INCOMPLETE'}</strong><span>MaleCNS 柱映射：Tm9 532266 → [{v7Status.evidence_boundaries.Tm9_official_synapse_coordinate.join(', ')}]；CT1 Lo1 逐突触列{v7Status.evidence_boundaries.CT1_per_synapse_Lo1_columnar_retinotopy_available ? '可用' : '不可用'}，全官方 LO 覆盖{v7Status.evidence_boundaries.CT1_complete_official_LO_column_coverage ? '完整' : '不完整'}</span><span>Mi4/C3 直接数值电压：{v7Status.evidence_boundaries.Mi4_C3_direct_numeric_voltage_candidates.join(' / ') || '无'}；独立候选 {v7Status.evidence_boundaries.Mi4_C3_independent_numeric_voltage_candidate_count}</span><span>CT1 电压候选：审计 {v7Status.evidence_boundaries.CT1_audited_candidate_count}；2025–2026 新候选 {v7Status.evidence_boundaries.CT1_incremental_2025_2026_candidate_count}；直接实验电压{v7Status.evidence_boundaries.CT1_direct_experimental_voltage_candidate_found ? '命中' : '未命中'}</span><span>T5 字段：flash {v7Status.evidence_boundaries.T5_voltage_field_counts.aggregated_full_field_OFF_flash}/15 · white-noise {v7Status.evidence_boundaries.T5_voltage_field_counts.raw_white_noise}/15 · grating {v7Status.evidence_boundaries.T5_voltage_field_counts.raw_drifting_grating}/15</span><span>Motyxia2 历史：{v7Status.evidence_boundaries.Motyxia2_public_history_branch_count} branches / {v7Status.evidence_boundaries.Motyxia2_public_history_commit_count} commits；逐记录日志{v7Status.evidence_boundaries.T5_record_log_found_in_Motyxia2_public_history ? '已找到' : '未命中'}</span><span>外部索引：linked log {v7Status.evidence_boundaries.T5_external_successful_indexes_linked_log_found ? '命中' : '未命中'}；publisher supplements {v7Status.evidence_boundaries.T5_publisher_supplements_inspected ? '已检查' : '未检查'} / record log {v7Status.evidence_boundaries.T5_publisher_supplements_contain_record_log ? '命中' : '未命中'}；PMC endpoint {v7Status.evidence_boundaries.T5_PMC_supplement_content_inspected ? '已检查' : '不可访问'}；Figshare {v7Status.evidence_boundaries.T5_Figshare_search_accessible ? '可访问' : '不可访问'}；全局不存在{v7Status.evidence_boundaries.T5_stimulus_log_global_absence_claimed ? '已声明' : '未声明'}</span><span>生成器默认值{v7Status.evidence_boundaries.T5_generator_defaults_used_as_record_fields ? '已代填' : '未代填'}</span></div>}<div className="contribution-grid"><span>上层规划：{v7Status?.contributions.upper_planner.status === 'paused' ? '暂停' : '未知'}</span><span>果蝇局部核：{v7Status?.contributions.fly_local_core.status === 'component_only_not_release_authorized' ? '组件证据，未获发布授权' : '未知'}</span><span>工程执行器：{v7Status?.contributions.engineering_executor.status === 'transparent_fixed_mapping_component_passed' ? '固定透明映射；v7 未接入' : '未知'}</span></div></section>
    {v7Status?.evidence_boundaries.Braun_calcium_fly_counts && <small className="v7-evidence-note">Braun 钙载荷：Tm2 {v7Status.evidence_boundaries.Braun_calcium_fly_counts.Tm2} flies · Tm9 {v7Status.evidence_boundaries.Braun_calcium_fly_counts.Tm9} · CT1 {v7Status.evidence_boundaries.Braun_calcium_fly_counts.CT1}；条件网格{v7Status.evidence_boundaries.Braun_calcium_condition_grids_complete ? '完整' : '不完整'}；合规电压源 {v7Status.evidence_boundaries.Braun_calcium_allowed_voltage_sources?.length ?? 0}</small>}
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
      <div><small>控制模式</small><strong>{controlMode === 'neural' ? 'MaleCNS v6' : '工程基线 v5'}</strong></div>
      {controlMode === 'neural' && <div><small>DNp20 运动适应</small><strong>{fmt((state?.motor.neural_adapter?.adaptation_rate ?? 0) * 100, 0)}% / step</strong></div>}
      {controlMode === 'neural' && <div><small>神经感觉档位</small><strong>{state?.sensory_profile === 'front' ? `前视 ${fmt(state?.retina.horizontal_fov_degrees ?? 0, 0)}°` : state?.sensory_profile ?? '—'}</strong></div>}
    </div>
    <div className="driving-controls"><label>控制模式 <select aria-label="控制模式" value={controlMode} disabled={running} onChange={e => onControlMode(e.target.value as 'assisted' | 'neural')}><option value="assisted">工程避障基线（v5）</option><option value="neural">纯 MaleCNS 决策实验</option></select></label><label><input type="checkbox" checked={learning} onChange={e => onLearning(e.target.checked)} /> 在线可塑性（实验）</label><label><input type="checkbox" checked={explore} onChange={e => onExplore(e.target.checked)} /> 探索噪声</label><label><input type="checkbox" checked={safetyConstraints} disabled={controlMode === 'neural'} onChange={e => onSafetyConstraints(e.target.checked)} /> 道路安全约束</label><button onClick={onRun}>{running ? '暂停' : '连续运行'}</button><button disabled={running} onClick={onStep}>单步</button><button disabled={running} onClick={() => onReset(true)}>新场景</button><button disabled={running} onClick={() => onReset(false)}>恢复发布策略</button></div>
    <p className="scientific-note">{controlMode === 'neural' ? state?.policy_checkpoint.loaded ? '已加载发布态 MaleCNS v6，默认服务与 v5/v6 检查点未改变。v7 仅离线实验：R1–R6 神经闭环、EPG/PEN/PEG 航向环和 FC2/PFL3/DNa02 透明链已有组件级实测与因果对照，但新布局的留一镜像对验证失败；严格 T4/T5 方向与 ON/OFF、LPLC1/LPLC2/LC4 分型也未通过。v7 未部署、未进入 MB 学习或外部 final。' : '纯神经决策：DNp20/DNpe017/MDN 输出直接映射为车辆动作；障碍、道路和规则仅作为视觉刺激、奖励与结果反馈。' : state?.policy_checkpoint.loaded ? `工程避障基线：已加载 ${state.policy_checkpoint.kind === 'frozen_calibrated' ? '冻结校准' : '实验学习'}策略。` : state?.policy_checkpoint.rejection ? '旧检查点已停用，当前策略未校准。' : '当前为未训练策略。'}</p>
  </section>
}
