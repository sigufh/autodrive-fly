import { render, screen } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import App from './App'

vi.mock('./api', () => ({
  fetchOverview: vi.fn(() => new Promise(() => undefined)),
  fetchPathways: vi.fn(() => new Promise(() => undefined)),
  fetchSkeleton: vi.fn(() => new Promise(() => undefined)),
  fetchDrivingState: vi.fn(() => new Promise(() => undefined)),
  resetDriving: vi.fn(), stepDriving: vi.fn(), streamDriving: vi.fn(),
}))

test('renders connectome and visual driving workspaces', () => {
  render(<App />)
  expect(screen.getByLabelText('MaleCNS 三维神经元')).toBeInTheDocument()
  expect(screen.getByLabelText('果蝇视觉驾驶')).toBeInTheDocument()
  expect(screen.getByText('完整拓扑图')).toBeInTheDocument()
  expect(screen.getByText('在线可塑性（实验）')).toBeInTheDocument()
  expect(screen.getByLabelText('在线可塑性（实验）')).not.toBeChecked()
  expect(screen.getByLabelText('探索噪声')).not.toBeChecked()
  expect(screen.getByLabelText('道路安全约束')).toBeChecked()
  expect(screen.getByLabelText('控制模式')).toHaveValue('assisted')
  expect(screen.queryByText('情绪解码')).not.toBeInTheDocument()
})
