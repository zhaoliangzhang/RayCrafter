"""Persist the architecture, coordinate gauge and injection contract."""
import hashlib
import json
from config import ROOT, RAMP_STEPS, VARIANT

CAMERA_CONTRACT = {
 'version': 1, 'camera_injection': 'concat_' + ('scheduled_' if RAMP_STEPS else '') + VARIANT,
 'reference_count': 2, 'query_count': 3, 'input_order': 'ref1,ref2,q1,q2,q3',
 'global_coordinate_frame': 'third_query_camera_full_rotation_and_translation',
 'sort_refs_by_distance': False,
 'dit_input_channels': 420, 'dit_image_channels': 36, 'dit_camera_channels': 384,
 'concat_pose_normalization': 'third_query_frame_mean_camera_distance_unchanged',
 'concat_perspective_ray_model': 'inverse_opencv_k1_k2_p1_p2_unchanged',
 'new_attention_perspective_ray_model': 'RayCrafter_pinhole_K_only',
 'temporal_rope': 'zero_phase',
 'zero_gate': 'hard_bypass_original_concat_attention_no_new_module_RNG',
 'inherited_training': 'standard_LoRA32_plus_patch_embedding_continues_training',
 'injection_schedule': 'linear_ramp' if RAMP_STEPS else 'none_lambda_1_from_first_update',
}
if VARIANT == 'scope_ff':
    CAMERA_CONTRACT.update(scope_pose_normalization='third_query_frame_translation_divided_by_scene_near_depth',
        scope_dir_freqs=8, scope_patch_size=True, scope_mlp_hidden=64,
        scope_output_initialization='zero_final_projection_random_hidden',
        wan_spatial_rope='retained_at_all_gate_values',
        injection='Q_rope+lambda*scope_Q; K_rope+lambda*scope_K; V_unchanged')
else:
    CAMERA_CONTRACT.update(rayrope_pose_normalization='third_query_frame_unit_reference_baseline',
        rayrope_internal_attention='per_camera_relative_geometry_not_sample_recentering',
        rayrope_apply_vo=True, rayrope_equirect_depth='signed_z_over_w',
        depth_head_lr=4e-4, depth_head_weight_decay=.5,
        wan_spatial_rope='baseline_branch_only',
        injection='(1-lambda)*original_spatial_rope_attention+lambda*RayRoPE_attention_before_shared_output_projection')

def contract_json():
    return json.dumps(CAMERA_CONTRACT, sort_keys=True, separators=(',', ':'))

def check_contract(metadata):
    if not metadata or metadata.get('camera_contract') != contract_json():
        raise ValueError('Checkpoint does not match this scheduled camera architecture')

def verify_vendor():
    provenance = json.loads((ROOT/'provenance/source.json').read_text())
    for name, expected in provenance['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Pinned RayCrafter file differs: {name}')
