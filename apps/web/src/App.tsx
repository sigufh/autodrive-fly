import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchDrivingState, fetchOverview, fetchPathways, fetchSkeleton, resetDriving, stepDriving, streamDriving } from './api'
import { CnsViewer } from './components/CnsViewer'
import { DrivingPanel } from './components/DrivingPanel'
import type { CnsOverview, DrivingState, PathwayOverview, SkeletonResponse } from './types'
import './styles.css'

export default function App() {
  const [skeleton, setSkeleton] = useState<SkeletonResponse>()
  const [overview, setOverview] = useState<CnsOverview>()
  const [pathways, setPathways] = useState<PathwayOverview>()
  const [driving, setDriving] = useState<DrivingState>()
  const [loading, setLoading] = useState(true), [running, setRunning] = useState(false)
  const [learning, setLearning] = useState(false), [explore, setExplore] = useState(false)
  const [safetyConstraints, setSafetyConstraints] = useState(true)
  const [controlMode, setControlMode] = useState<'assisted' | 'neural'>('assisted')
  const [error, setError] = useState('')
  const generation = useRef(0), run = useRef<AbortController | null>(null)
  const selectNeuron = useCallback((bodyId: number) => {
    const request = ++generation.current; setLoading(true)
    fetchSkeleton(bodyId).then(s => { if (generation.current === request) setSkeleton(s) })
      .catch(e => setError(String(e))).finally(() => { if (generation.current === request) setLoading(false) })
  }, [])
  useEffect(() => {
    let live = true
    Promise.all([fetchOverview(), fetchPathways(), fetchDrivingState()]).then(async ([o, p, d]) => {
      const drivingState = d.scenario === 'highway' ? d : await resetDriving(0, false, 'highway', 'assisted')
      if (live) { setOverview(o); setPathways(p); setDriving(drivingState) }
    }).catch(e => { if (live) setError(String(e)) })
    fetchSkeleton(10059).then(s => { if (live) setSkeleton(s) })
      .catch(e => { if (live) setError(String(e)) }).finally(() => { if (live) setLoading(false) })
    return () => { live = false; run.current?.abort() }
  }, [])
  async function toggleRun() {
    if (running) { run.current?.abort(); setRunning(false); return }
    const controller = new AbortController(); run.current = controller; setRunning(true); setError('')
    try { await streamDriving(learning, explore, safetyConstraints, controlMode, state => setDriving(state), controller.signal) }
    catch (reason) { if (!controller.signal.aborted) setError(reason instanceof Error ? reason.message : '仿真失败') }
    finally { setRunning(false) }
  }
  async function step() { try { setDriving(await stepDriving(1, learning, explore, safetyConstraints, controlMode)) } catch (reason) { setError(String(reason)) } }
  async function reset(keep: boolean, targetControlMode = controlMode) {
    try { setDriving(await resetDriving(Math.floor(Math.random() * 1_000_000), keep, 'highway', targetControlMode)) }
    catch (reason) { setError(String(reason)) }
  }
  return <main>
    <div className="brain-workspace">
      {error && <p role="alert" className="error global-error">{error}</p>}
      <CnsViewer overview={overview} pathways={pathways} skeleton={skeleton} loading={loading} activity={driving?.activity} phase={running ? 'closed-loop' : 'paused'} onSelect={selectNeuron} />
    </div>
    <DrivingPanel state={driving} running={running} learning={learning} explore={explore} safetyConstraints={safetyConstraints} controlMode={controlMode} onControlMode={value => { setControlMode(value); reset(false, value) }} onLearning={setLearning} onExplore={setExplore} onSafetyConstraints={setSafetyConstraints} onRun={toggleRun} onStep={step} onReset={reset} />
  </main>
}
