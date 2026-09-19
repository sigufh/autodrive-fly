import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import App from './App'
import { fetchV7Status } from './api'

vi.mock('./api', () => ({
  fetchOverview: vi.fn(() => new Promise(() => undefined)),
  fetchPathways: vi.fn(() => new Promise(() => undefined)),
  fetchSkeleton: vi.fn(() => new Promise(() => undefined)),
  fetchDrivingState: vi.fn(() => new Promise(() => undefined)),
  fetchV7Status: vi.fn(() => new Promise(() => undefined)),
  resetDriving: vi.fn(), stepDriving: vi.fn(), streamDriving: vi.fn(),
}))

afterEach(() => {
  cleanup()
  vi.clearAllMocks()
})

test('renders connectome and visual driving workspaces', () => {
  render(<App />)
  expect(screen.getByLabelText('MaleCNS 三维神经元')).toBeInTheDocument()
  expect(screen.getByLabelText('果蝇视觉驾驶')).toBeInTheDocument()
  expect(screen.getByLabelText('实时因果链')).toBeInTheDocument()
  expect(screen.getByLabelText('v7 离线验证状态')).toBeInTheDocument()
  expect(screen.getByText(/当前运行链 · assisted-v5/)).toBeInTheDocument()
  expect(screen.getByText(/果蝇局部核：未知/)).toBeInTheDocument()
  expect(screen.getByText('工程辅助')).toBeInTheDocument()
  expect(within(screen.getByLabelText('v7 离线验证状态')).getByText('读取中')).toBeInTheDocument()
  expect(screen.getByText('完整拓扑图')).toBeInTheDocument()
  expect(screen.getByText('在线可塑性（实验）')).toBeInTheDocument()
  expect(screen.getByLabelText('在线可塑性（实验）')).not.toBeChecked()
  expect(screen.getByLabelText('探索噪声')).not.toBeChecked()
  expect(screen.getByLabelText('道路安全约束')).toBeChecked()
  expect(screen.getByLabelText('控制模式')).toHaveValue('assisted')
  expect(screen.queryByText('情绪解码')).not.toBeInTheDocument()
})

test('renders hash-verified offline v7 gates without presenting v7 as runtime', async () => {
  vi.mocked(fetchV7Status).mockResolvedValueOnce({
    version: 'v7-experimental', source: 'hash-verified-offline-goal-audit', current_stage: 'controlled_vision', objective_complete: false, deployment_enabled: false, default_runtime_changed: false, audit_sha256: 'a'.repeat(64),
    gates: { T4_T5_direction_and_ON_OFF: false, LPLC1_near_collision: false, LPLC2_radial_opponency: false, LC4_angular_speed: false, EPG_PEN_PEG_heading: true, PFL3_DNa_transparent_mapping: true, causal_visual_navigation: false, external_final: false },
    evidence_boundaries: { nine_source_contract_complete: false, Mi4_C3_direct_numeric_voltage_candidates: ['Groschner_2022'], Mi4_C3_independent_numeric_voltage_candidate_count: 0, T5_voltage_field_counts: { aggregated_full_field_OFF_flash: 7, raw_white_noise: 8, raw_drifting_grating: 9 }, Braun_calcium_fly_counts: { Tm2: 9, Tm9: 11, CT1: 9 }, Braun_calcium_condition_grids_complete: true, Braun_calcium_allowed_voltage_sources: [], Gou_Dryad_archive_hash_locally_verified: true, Gou_Dryad_local_processed_calcium_sources: ['Mi1', 'Tm3', 'Tm1', 'Tm2'], Gou_Dryad_flash_fly_axis_sizes: { Mi1: 16, Tm3: 11, Tm1: 9, Tm2: 8 }, Gou_Dryad_moving_bar_fly_axis_sizes: { Mi1: 10, Tm3: 6, Tm1: 8, Tm2: 12 }, Gou_Dryad_stable_biological_individual_IDs_verified: false, Gou_DANDI_asset_level_stable_participant_IDs_verified: true, Gou_DANDI_unique_subject_ID_count: 282, Gou_Dryad_distinct_fliesUsed_label_count: 66, Gou_Dryad_DANDI_identity_crosswalk_match_count: 0, Gou_Dryad_rows_to_DANDI_subject_crosswalk_verified: false, Gou_Dryad_experimental_membrane_voltage: false, v7_offline_time_coordinate_contract_complete: true, v7_offline_horizontal_coordinate_contract_complete: true, v7_offline_two_dimensional_angular_calibration_complete: false, v7_offline_frame_interval_milliseconds: 10, v7_offline_substep_interval_milliseconds: 2.5, v7_horizontal_fov_degrees: 143.2394487827058, T4_source_mapping_mode: 'exact_type_average', T4_source_recording_level_body_assignment: false, T4_exact_type_average_mapping_complete: true, T4_author_minmax_formula_reproduced: true, T4_state_mapping_held_out_outside_fraction_by_source: { Mi1: 0.1295625, Tm3: 0.17478125, Mi4: 0.2870446428571429, C3: 0.18847916666666667 }, T4_author_minmax_semantics_match_v7_state: false, T4_millivolts_to_v7_state_mapping_available: false, T5_record_specific_stimulus_logs_available: false, Motyxia2_public_history_branch_count: 22, Motyxia2_public_history_commit_count: 447, T5_record_log_found_in_Motyxia2_public_history: false, T5_external_successful_indexes_linked_log_found: false, T5_PMC_supplement_content_inspected: false, T5_publisher_supplements_inspected: true, T5_publisher_supplements_contain_record_log: false, T5_Figshare_search_accessible: false, T5_stimulus_log_global_absence_claimed: false, T5_generator_defaults_used_as_record_fields: false, T5_stimulus_provenance_complete: false, CT1_audited_candidate_count: 11, CT1_incremental_2025_2026_candidate_count: 3, CT1_direct_experimental_voltage_candidate_found: false, CT1_PuRe_archive_contents_verified: true, CT1_PuRe_new_numerical_payload_verified: false, Tm9_official_synapse_coordinate: [15, 2], CT1_per_synapse_Lo1_columnar_retinotopy_available: true, CT1_complete_official_LO_column_coverage: false },
    contributions: { upper_planner: { status: 'paused', active_in_default_runtime: false }, fly_local_core: { status: 'component_only_not_release_authorized', active_v7_in_default_runtime: false }, engineering_executor: { status: 'transparent_fixed_mapping_component_passed', v7_deployment_enabled: false } },
  })
  render(<App />)
  expect(await screen.findByText('T4_T5_direction_and_ON_OFF · STOP')).toBeInTheDocument()
  expect(screen.getByText('EPG_PEN_PEG_heading · PASS')).toBeInTheDocument()
  expect(screen.getByText(/当前阶段：controlled_vision · 默认服务未改变/)).toBeInTheDocument()
  expect(screen.getByText('九源合同：INCOMPLETE')).toBeInTheDocument()
  expect(screen.getByText('Mi4/C3 直接数值电压：Groschner_2022；独立候选 0')).toBeInTheDocument()
  expect(screen.getByText('CT1 电压候选：审计 11；2025–2026 新候选 3；直接实验电压未命中')).toBeInTheDocument()
  expect(screen.getByText('PuRe ZIP：已核验；新增数值载荷未命中')).toBeInTheDocument()
  expect(screen.getByText('MaleCNS 柱映射：Tm9 532266 → [15, 2]；CT1 Lo1 逐突触列可用，全官方 LO 覆盖不完整')).toBeInTheDocument()
  expect(screen.getByText('T5 字段：flash 7/15 · white-noise 8/15 · grating 9/15')).toBeInTheDocument()
  expect(screen.getByText('Braun 钙载荷：Tm2 9 flies · Tm9 11 · CT1 9；条件网格完整；合规电压源 0')).toBeInTheDocument()
  expect(screen.getByText(/离线刺激坐标：10 ms\/frame · 2.5 ms\/substep · 横向 FOV 143.24°/)).toBeInTheDocument()
  expect(screen.getByText(/T4 source→MaleCNS：exact_type_average 四源完整/)).toBeInTheDocument()
  const stateMapping = screen.getByText((_content, element) => element?.tagName === 'SMALL' && Boolean(element.textContent?.includes('held-out 越界率 Mi1/Tm3/Mi4/C3 = 13.0%/17.5%/28.7%/18.8%')))
  expect(stateMapping).toHaveTextContent('T4 mV→state：作者全 cohort min–max 已复现')
  expect(stateMapping).toHaveTextContent('与 v7 signed-tanh 状态语义不一致，映射门未通过')
  expect(screen.getByText(/Gou ↔ DANDI 身份：DANDI 资产级 participant 282 个唯一 ID/)).toBeInTheDocument()
  expect(screen.getByText(/离线刺激坐标：10 ms\/frame · 2.5 ms\/substep · 横向 FOV 143.24°/)).toBeInTheDocument()
  expect(screen.getByText('Motyxia2 历史：22 branches / 447 commits；逐记录日志未命中')).toBeInTheDocument()
  expect(
    screen.getByText(
      /外部索引：linked log 未命中；publisher supplements 已检查 \/ record log 未命中/,
    ),
  ).toBeInTheDocument()
  expect(screen.getByText(/PMC endpoint 不可访问/)).toBeInTheDocument()
  expect(screen.getByText('生成器默认值未代填')).toBeInTheDocument()
  expect(within(screen.getByLabelText('v7 离线验证状态')).getByText('未部署')).toBeInTheDocument()
})
