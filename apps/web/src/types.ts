export type Vec3 = [number, number, number]
export type SkeletonResponse = { body_id: number; units: string; source_vertices: number; source_edges: number; display_edges: number; segments: [Vec3, Vec3][] }
export type CnsOverview = { release: string; canonical_nodes: number; positioned_nodes: number; units: string; body_ids: number[]; positions: Vec3[]; class_ids: number[]; classes: string[] }
export type Pathway = { source?: number; target?: number; pre?: number; post?: number; edge_count?: number; synapse_weight?: number; weight?: number; from: Vec3 | null; to: Vec3 | null; positioned?: boolean }
export type PathwayOverview = {
  release: string; all_edges_accounted: number; all_synapse_weight_accounted: number;
  bundled_paths: Pathway[]; strong_paths: Pathway[]; strong_path_policy: string;
  topology_nodes: { id: number; name: string; neurons: number; positioned: boolean }[];
  exported_bundle_edges: number; exported_bundle_weight: number;
}
export type NeuronActivity = { body_id: number; value: number }
export type EdgeActivity = { pre: number; post: number; value: number }
export type ActivityFrame = {
  type: 'activity'; step: number; total_steps: number; elapsed_ms: number;
  units: string; neurons: NeuronActivity[]; pathways: EdgeActivity[]; group_activity?: Record<string, number>;
  statistics: { all_nodes: number; active_nodes: number; positioned_nodes: number; unpositioned_active_nodes: number; displayed_nodes: number; displayed_edges: number; max_abs_state: number; display_max_abs_state: number }
}
export type DrivingState = {
  scenario: 'highway' | 'city';
  control_mode: 'assisted' | 'neural';
  sensory_profile: 'front' | 'panorama' | 'panorama_flow' | 'panorama_flow_body';
  environment: {
    road_half_width: number; road_length: number; pair_seed: number; mirror: number;
    vehicle: { x: number; y: number; heading: number; speed: number; steering: number; world_x?: number; world_y?: number; world_heading?: number };
    obstacles: { x: number; y: number; radius: number }[]; sensor_rays: number[]; obstacle_rays: number[]; wall_rays: number[];
    trajectory: [number, number][]; projected_trajectory: [number, number][];
    step: number; done: boolean; success: boolean; total_reward: number; terminal_reason: string | null; obstacles_passed: number; first_obstacle_passed: boolean; first_obstacle_side: 'left' | 'right' | null;
    city?: { scenario: string; name: string; map_bounds: [number, number, number, number]; centerline: [number, number][]; world_trajectory: [number, number][]; actors: { id: number; x: number; y: number; route_progress: number; lane_offset: number; radius: number }[]; road: string; next_maneuver: string; speed_limit_mps: number; traffic_light: 'red' | 'green'; stop_line_progress: number | null; distance_to_stop_line: number | null; rule_status: string; violations: string[]; intersections: { name: string; progress: number; signal: 'red' | 'green' }[] };
  };
  action: { steering: number; throttle: number; reverse: number; drive: number }; reward: number; safety_signal: number; learning: boolean; sensory_learning?: boolean; elapsed_ms?: number;
  raw_action: { steering: number; throttle: number; reverse: number; drive: number };
  lane_constraint: { active: boolean; blend: number; correction: number; neural_steering?: number; visual_avoidance?: number; road_recovery?: number; route_steering?: number };
  control_statistics: { mean_abs_steering: number; mean_abs_steering_change: number; far_mean_abs_steering: number; far_steps: number; steering_sign_changes: number; max_abs_lateral: number; constraint_rate: number; mean_abs_constraint: number };
  policy_checkpoint: { loaded: boolean; path: string; kind: string; rejection: string | null };
  dopamine: { rule: string; dopamine: number; lateral_dopamine: [number, number]; plastic_synapses: number; changed_synapses: number; mean_gain: number; min_gain: number; max_gain: number; updates: number };
  retina: { mapped_receptors: number; source_type: string; mapping: string; width: number; height: number; stimulus: number[][]; neural_stimulus_width?: number; horizontal_fov_degrees?: number };
  motor: { body_ids: number[]; names: string[]; mapping: string; brain_substeps_per_action: number; neural_adapter?: { type: string; steering_gain: number; adaptation_rate: number; steering_baseline: number }; sensory_projection?: { profile: string; last_flow: number; last_flow_regions?: number[]; last_yaw_rate: number; T4_T5_horizontal_flow: number; T4_T5_spatially_mapped?: number; T4_T5_spatially_unmapped?: number; haltere_yaw_rate: number; ascending_proprioception: number; haltere_by_side: [number, number]; proprioception_by_side: [number, number]; body_mapped_unique?: number; body_unmapped?: number; pathway_plasticity?: { rule: string; group_names: string[]; group_gains: number[]; unique_source_neurons: number; existing_outgoing_synapses: number; updates: number } } };
  dopamine_neurons: { body_ids: number[]; type: string; signal: string };
  activity: ActivityFrame;
}
export type DrivingEvent = ({ type: 'driving_state' } & DrivingState) | { type: 'done' } | { type: 'error'; message: string }
export type V7Status = {
  version: 'v7-experimental'; source: 'hash-verified-offline-goal-audit'; current_stage: string; objective_complete: boolean; deployment_enabled: boolean; default_runtime_changed: boolean; audit_sha256: string;
  gates: { T4_T5_direction_and_ON_OFF: boolean; LPLC1_near_collision: boolean; LPLC2_radial_opponency: boolean; LC4_angular_speed: boolean; EPG_PEN_PEG_heading: boolean; PFL3_DNa_transparent_mapping: boolean; causal_visual_navigation: boolean; external_final: boolean };
  evidence_boundaries: {
    nine_source_contract_complete: boolean;
    Mi4_C3_direct_numeric_voltage_candidates: string[];
    Mi4_C3_independent_numeric_voltage_candidate_count: number;
    T5_voltage_field_counts: { aggregated_full_field_OFF_flash: number; raw_white_noise: number; raw_drifting_grating: number };
    Braun_calcium_fly_counts?: { Tm2: number; Tm9: number; CT1: number };
    Braun_calcium_condition_grids_complete?: boolean;
    Braun_calcium_allowed_voltage_sources?: string[];
    T5_record_specific_stimulus_logs_available: boolean;
    Motyxia2_public_history_branch_count: number;
    Motyxia2_public_history_commit_count: number;
    T5_record_log_found_in_Motyxia2_public_history: boolean;
    T5_external_successful_indexes_linked_log_found: boolean;
    T5_PMC_supplement_content_inspected: boolean;
    T5_publisher_supplements_inspected: boolean;
    T5_publisher_supplements_contain_record_log: boolean;
    T5_Figshare_search_accessible: boolean;
    T5_stimulus_log_global_absence_claimed: boolean;
    T5_generator_defaults_used_as_record_fields: boolean;
    T5_stimulus_provenance_complete: boolean;
    CT1_audited_candidate_count: number;
    CT1_incremental_2025_2026_candidate_count: number;
    CT1_direct_experimental_voltage_candidate_found: boolean;
    CT1_PuRe_archive_contents_verified: boolean;
    CT1_PuRe_new_numerical_payload_verified: boolean;
    Tm9_official_synapse_coordinate: [number, number];
    CT1_per_synapse_Lo1_columnar_retinotopy_available: boolean;
    CT1_complete_official_LO_column_coverage: boolean;
  };
  contributions: { upper_planner: { status: string; active_in_default_runtime: boolean }; fly_local_core: { status: string; active_v7_in_default_runtime: boolean }; engineering_executor: { status: string; v7_deployment_enabled: boolean } };
}
