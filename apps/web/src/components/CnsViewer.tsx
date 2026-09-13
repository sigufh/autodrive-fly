import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { ActivityFrame, CnsOverview, PathwayOverview, SkeletonResponse, Vec3 } from '../types'

type Props = { overview?: CnsOverview; pathways?: PathwayOverview; skeleton?: SkeletonResponse; loading: boolean; activity?: ActivityFrame; phase: string; onSelect: (id: number) => void }
const POS = new THREE.Color('#ffc857'), NEG = new THREE.Color('#49c6ff')
export function stateColor(value: number, scale: number) {
  const strength = Math.cbrt(Math.min(1, Math.abs(value) / Math.max(scale, 1e-12)))
  const c = (value >= 0 ? POS : NEG).clone().multiplyScalar(0.2 + 0.8 * strength)
  return [c.r, c.g, c.b]
}
export function topologyCurve(a: { x: number; y: number }, b: { x: number; y: number }, self: boolean) {
  if (self) return `M ${a.x - 7} ${a.y - 7} C ${a.x - 58} ${a.y - 75}, ${a.x + 58} ${a.y - 75}, ${a.x + 7} ${a.y - 7}`
  const dx = b.x - a.x, dy = b.y - a.y, length = Math.hypot(dx, dy)
  const ux = dx / length, uy = dy / length
  return `M ${a.x + ux * 12} ${a.y + uy * 12} Q ${(a.x + b.x) / 2 - uy * 28} ${(a.y + b.y) / 2 + ux * 28}, ${b.x - ux * 16} ${b.y - uy * 16}`
}
function Topology({ pathways, activity }: { pathways: PathwayOverview; activity?: ActivityFrame }) {
  const [selected, setSelected] = useState<number | null>(null)
  const nodes = pathways.topology_nodes
  const groupScale = Math.max(1e-12, ...Object.values(activity?.group_activity ?? {}).map(Math.abs))
  const locations = nodes.map((_, i) => ({ x: 490 + 335 * Math.cos(2 * Math.PI * i / nodes.length), y: 360 + 265 * Math.sin(2 * Math.PI * i / nodes.length) }))
  return <div className="topology-view">
    <p>完整有向拓扑示意 · 非解剖坐标 · {pathways.bundled_paths.length} 组连接 · {pathways.exported_bundle_edges.toLocaleString()} 条原始边</p>
    <button onClick={() => setSelected(null)}>显示全部组</button>
    <svg viewBox="0 0 980 730" aria-label="全通路拓扑图" data-edge-coverage={pathways.exported_bundle_edges}>
      <defs><marker id="direction" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#b3e9d7" /></marker></defs>
      {pathways.bundled_paths.map(p => {
        const source = p.source!, target = p.target!
        const visible = selected === null || source === selected || target === selected
        return <path key={`${source}-${target}`} data-source={source} data-target={target} d={topologyCurve(locations[source], locations[target], source === target)} fill="none" stroke={source === target ? '#ffce76' : '#71bcaa'} opacity={visible ? 0.55 : 0.025} strokeWidth={0.7 + Math.log10(1 + (p.synapse_weight ?? 0)) / 3} markerEnd="url(#direction)"><title>{nodes[source].name} → {nodes[target].name}: {p.edge_count?.toLocaleString()} 条边 / {p.synapse_weight?.toLocaleString()} 突触权重</title></path>
      })}
      {nodes.map((node, i) => <g key={node.id} onClick={() => setSelected(node.id)} role="button" tabIndex={0} aria-label={node.name} onKeyDown={e => { if (e.key === 'Enter') setSelected(node.id) }}>
        <circle cx={locations[i].x} cy={locations[i].y} r={11} fill={activity?.group_activity ? new THREE.Color(...stateColor(activity.group_activity[node.name] ?? 0, groupScale) as [number, number, number]).getStyle() : node.positioned ? '#79dfb4' : '#ffcc83'} stroke="#effbf6" />
        <text x={locations[i].x + (locations[i].x < 490 ? -16 : 16)} y={locations[i].y + 4} textAnchor={locations[i].x < 490 ? 'end' : 'start'} fill="#e8f8f1" fontSize="10">{node.name}</text>
        <title>{node.neurons} 神经元 · {node.positioned ? '部分有解剖坐标' : '无解剖坐标，仅拓扑位置'}</title>
      </g>)}
    </svg>
  </div>
}

export function CnsViewer({ overview, pathways, skeleton, loading, activity, phase, onSelect }: Props) {
  const host = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<{ renderer: THREE.WebGLRenderer; camera: THREE.PerspectiveCamera; skeleton: THREE.LineSegments; strong: THREE.LineSegments; glow: THREE.Points; active: THREE.LineSegments; center: THREE.Vector3; positions: Float32Array; index: Map<number, number> } | null>(null)
  const [mode, setMode] = useState<'anatomy' | 'topology'>('anatomy')
  const [showStrong, setShowStrong] = useState(false)
  const [renderError, setRenderError] = useState('')
  const [bodyInput, setBodyInput] = useState('10059')
  const [revision, setRevision] = useState(0)
  const visibleRef = useRef(showStrong)
  visibleRef.current = showStrong

  useEffect(() => {
    if (!host.current || !overview || !pathways) return
    const element = host.current
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true }) } catch { setRenderError('WebGL 不可用，请使用完整拓扑图。'); return }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2))
    renderer.setClearColor('#102821')
    element.appendChild(renderer.domElement)
    const scene = new THREE.Scene(), root = new THREE.Group()
    scene.add(root)
    root.rotation.x = -Math.PI / 2
    const camera = new THREE.PerspectiveCamera(40, 1, 1, 10000)
    const pointGeometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(overview.positions.flat(), 3))
    pointGeometry.computeBoundingBox()
    const center = pointGeometry.boundingBox!.getCenter(new THREE.Vector3())
    pointGeometry.translate(-center.x, -center.y, -center.z)
    pointGeometry.computeBoundingSphere()
    const radius = pointGeometry.boundingSphere!.radius
    camera.position.set(radius * 0.15, radius * 0.1, radius * 2.75)
    camera.lookAt(0, 0, 0)
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.enableDamping = true
    controls.minDistance = radius * 0.4; controls.maxDistance = radius * 6
    const points = new THREE.Points(pointGeometry, new THREE.PointsMaterial({ size: 1.5, color: '#81dcba', opacity: 0.72, transparent: true, depthWrite: false }))
    root.add(points)
    const lines = pathways.strong_paths.flatMap(p => p.from && p.to ? [...p.from, ...p.to] : [])
    const strongGeometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(lines, 3))
    strongGeometry.translate(-center.x, -center.y, -center.z)
    const strong = new THREE.LineSegments(strongGeometry, new THREE.LineBasicMaterial({ color: '#68cda9', opacity: 0.12, transparent: true, depthWrite: false }))
    strong.visible = visibleRef.current
    root.add(strong)
    const selected = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: '#ffeda2', depthTest: false }))
    selected.renderOrder = 3; root.add(selected)
    const glow = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 6, vertexColors: true, depthTest: false, sizeAttenuation: true }))
    glow.frustumCulled = false; glow.renderOrder = 5; root.add(glow)
    const active = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false }))
    active.frustumCulled = false; active.renderOrder = 4; root.add(active)
    const index = new Map(overview.body_ids.map((id, i) => [id, i]))
    const positions = pointGeometry.getAttribute('position').array as Float32Array
    sceneRef.current = { renderer, camera, skeleton: selected, strong, glow, active, center, positions, index }
    const raycaster = new THREE.Raycaster(); raycaster.params.Points!.threshold = radius * 0.004
    const select = (e: MouseEvent) => {
      const box = renderer.domElement.getBoundingClientRect()
      raycaster.setFromCamera(new THREE.Vector2((e.clientX - box.left) / box.width * 2 - 1, -(e.clientY - box.top) / box.height * 2 + 1), camera)
      const hit = raycaster.intersectObject(points)[0]
      if (hit?.index !== undefined) onSelect(overview.body_ids[hit.index])
    }
    renderer.domElement.addEventListener('dblclick', select)
    const resize = () => { const w = element.clientWidth, h = element.clientHeight; if (!w || !h) return; camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h) }
    const observer = new ResizeObserver(resize); observer.observe(element); resize()
    let frame = 0
    const render = () => { controls.update(); renderer.domElement.dataset.cameraPosition = camera.position.toArray().map(v => v.toFixed(4)).join(','); renderer.render(scene, camera); frame = requestAnimationFrame(render) }
    render(); setRevision(v => v + 1)
    return () => {
      cancelAnimationFrame(frame); observer.disconnect(); controls.dispose(); renderer.domElement.removeEventListener('dblclick', select)
      for (const object of [points, strong, selected, glow, active]) { object.geometry.dispose(); (object.material as THREE.Material).dispose() }
      renderer.dispose(); renderer.domElement.remove(); sceneRef.current = null
    }
  }, [overview, pathways, onSelect])

  useEffect(() => {
    const scene = sceneRef.current
    if (!scene || !skeleton) return
    const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(skeleton.segments.flat(2), 3))
    geometry.translate(-scene.center.x, -scene.center.y, -scene.center.z)
    scene.skeleton.geometry.dispose(); scene.skeleton.geometry = geometry
  }, [skeleton, revision])
  useEffect(() => { if (sceneRef.current) sceneRef.current.strong.visible = showStrong }, [showStrong, revision])
  useEffect(() => {
    const scene = sceneRef.current
    if (!scene) return
    const { index, positions } = scene
    const position = (id: number): Vec3 | null => { const i = index.get(id); return i === undefined ? null : [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]] }
    const nodePositions: number[] = [], colors: number[] = []
    const scale = activity?.statistics.max_abs_state ?? 1
    for (const n of activity?.neurons ?? []) { const p = position(n.body_id); if (p) { nodePositions.push(...p); colors.push(...stateColor(n.value, scale)) } }
    const edgePositions: number[] = [], edgeColors: number[] = []
    const edgeScale = Math.max(1e-12, ...(activity?.pathways ?? []).map(p => Math.abs(p.value)))
    for (const e of activity?.pathways ?? []) {
      const a = position(e.pre), b = position(e.post)
      if (!a || !b) continue
      const from = new THREE.Vector3(...a), to = new THREE.Vector3(...b), d = to.clone().sub(from)
      const length = d.length(); if (length === 0) continue
      d.normalize(); const side = new THREE.Vector3(-d.y, d.x, 0); if (side.lengthSq() < 0.01) side.set(0, 1, 0); side.normalize()
      const arrow = Math.min(length * 0.2, 5), back = to.clone().addScaledVector(d, -arrow)
      const color = stateColor(e.value, edgeScale)
      for (const pair of [[from, to], [to, back.clone().addScaledVector(side, arrow * 0.5)], [to, back.clone().addScaledVector(side, -arrow * 0.5)]]) {
        edgePositions.push(...pair[0].toArray(), ...pair[1].toArray()); edgeColors.push(...color, ...color)
      }
    }
    const replace = (object: THREE.Points | THREE.LineSegments, p: number[], c: number[]) => { const geometry = new THREE.BufferGeometry().setAttribute('position', new THREE.Float32BufferAttribute(p, 3)).setAttribute('color', new THREE.Float32BufferAttribute(c, 3)); object.geometry.dispose(); object.geometry = geometry }
    replace(scene.glow, nodePositions, colors); replace(scene.active, edgePositions, edgeColors)
    scene.renderer.domElement.dataset.displayedNodes = String(nodePositions.length / 3)
    scene.renderer.domElement.dataset.displayedEdges = String(activity?.pathways.length ?? 0)
  }, [activity, revision])

  return <section className="viewer-panel" aria-label="MaleCNS 三维神经元">
    <div className="viewer-head"><div><span className="eyebrow">MALE CNS / V1.0</span><h1>视觉—运动闭环</h1><small>{overview?.canonical_nodes.toLocaleString() ?? '…'} neurons · {pathways?.all_edges_accounted.toLocaleString() ?? '…'} edges</small></div><span role="status">{phase}</span></div>
    <div className="viewer-controls"><button onClick={() => setMode('anatomy')} aria-pressed={mode === 'anatomy'}>解剖 3D</button><button onClick={() => setMode('topology')} aria-pressed={mode === 'topology'}>完整拓扑图</button><button onClick={() => setShowStrong(v => !v)} aria-pressed={showStrong}>40K 强连接</button><input aria-label="神经元 ID" value={bodyInput} onChange={e => setBodyInput(e.target.value)} /><button onClick={() => { const id = Number(bodyInput); if (Number.isSafeInteger(id) && id > 0) onSelect(id) }}>加载骨架</button></div>
    <div className="canvas-frame" style={{ display: mode === 'anatomy' ? 'block' : 'none' }}><div ref={host} className="canvas-host" />{loading && <div className="loading">加载真实解剖数据…</div>}{renderError && <p role="alert">{renderError}</p>}</div>
    {mode === 'topology' && pathways && <Topology pathways={pathways} activity={activity} />}
    <div className="state-legend"><span style={{ color: '#ffc857' }}>正活动</span><span style={{ color: '#49c6ff' }}>抑制性贡献</span><span>亮度 ∝ |状态| · 箭头 pre → post · 无毫伏单位</span></div>
    <div className="viewer-foot"><span>BODY {skeleton?.body_id ?? '—'}</span>{activity ? <span>驾驶步 {activity.step} · 全图活跃 {activity.statistics.active_nodes.toLocaleString()} · 缺坐标 {activity.statistics.unpositioned_active_nodes.toLocaleString()} · 已绘 {activity.neurons.length} 节点 / {activity.pathways.length} 边 · |x|max {activity.statistics.max_abs_state.toExponential(2)}</span> : <span>等待闭环状态；不是实测膜电位</span>}</div>
  </section>
}
