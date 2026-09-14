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
  environment: {
    road_half_width: number; road_length: number; pair_seed: number; mirror: number;
    vehicle: { x: number; y: number; heading: number; speed: number; steering: number; world_x?: number; world_y?: number; world_heading?: number };
    obstacles: { x: number; y: number; radius: number }[]; sensor_rays: number[]; obstacle_rays: number[]; wall_rays: number[];
    trajectory: [number, number][]; projected_trajectory: [number, number][];
    step: number; done: boolean; success: boolean; total_reward: number; terminal_reason: string | null; obstacles_passed: number; first_obstacle_passed: boolean; first_obstacle_side: 'left' | 'right' | null;
    city?: { scenario: string; name: string; map_bounds: [number, number, number, number]; centerline: [number, number][]; world_trajectory: [number, number][]; actors: { id: number; x: number; y: number; route_progress: number; lane_offset: number; radius: number }[]; road: string; next_maneuver: string; speed_limit_mps: number; traffic_light: 'red' | 'green'; stop_line_progress: number | null; distance_to_stop_line: number | null; rule_status: string; violations: string[]; intersections: { name: string; progress: number; signal: 'red' | 'green' }[] };
  };
  action: { steering: number; throttle: number; reverse: number; drive: number }; reward: number; safety_signal: number; learning: boolean; elapsed_ms?: number;
  raw_action: { steering: number; throttle: number; reverse: number; drive: number };
  lane_constraint: { active: boolean; blend: number; correction: number; neural_steering?: number; visual_avoidance?: number; road_recovery?: number; route_steering?: number };
  control_statistics: { mean_abs_steering: number; mean_abs_steering_change: number; far_mean_abs_steering: number; far_steps: number; steering_sign_changes: number; max_abs_lateral: number; constraint_rate: number; mean_abs_constraint: number };
  policy_checkpoint: { loaded: boolean; path: string; kind: string; rejection: string | null };
  dopamine: { rule: string; dopamine: number; lateral_dopamine: [number, number]; plastic_synapses: number; changed_synapses: number; mean_gain: number; min_gain: number; max_gain: number; updates: number };
  retina: { mapped_receptors: number; source_type: string; mapping: string; width: number; height: number; stimulus: number[][] };
  motor: { body_ids: number[]; names: string[]; mapping: string; brain_substeps_per_action: number; neural_adapter?: { type: string; steering_gain: number; adaptation_rate: number; steering_baseline: number } };
  dopamine_neurons: { body_ids: number[]; type: string; signal: string };
  activity: ActivityFrame;
}
export type DrivingEvent = ({ type: 'driving_state' } & DrivingState) | { type: 'done' } | { type: 'error'; message: string }
