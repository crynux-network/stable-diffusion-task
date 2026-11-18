# Video Task Arguments and Pipeline Compatibility

## Authority and scope

This document specifies `VideoTaskArgs` and the runtime execution contract inside `stable-diffusion-task` from validated task arguments to canonical encoded `.mp4` video artifacts.

`VideoTaskArgs` reuses the shared model-loading, `dtype` resolution, component-override (`components`), and task-time quantization (`quantization_config`) foundations of `ImageTaskArgs` specified in [`../model-compatibility.md`](../model-compatibility.md) and [`../quantization.md`](../quantization.md), while extending conditioning inputs, temporal generation controls, multi-LoRA routing, and modular pipeline execution to cover modern open-weight video and joint audio-video models in `diffusers`.

Public HTTP API schemas (such as OpenAI-compatible video endpoints, async job polling, or Bridge/AS adaptation) and Relay VSS score verification remain outside `stable-diffusion-task`.

The core execution path MUST NOT branch on model IDs, repository names, or hard-coded checkpoint allowlists. All behavior MUST be derived from repository metadata (`model_index.json` or `modular_model_index.json`), installed Diffusers pipeline/block classes, component configs, and Python call signatures. Unknown task fields MUST fail validation (`extra="forbid"`).

---

## 1. Covered Generation Modes

One unified `VideoTaskArgs` schema covers all eight production video generation modes identified in short-drama and general AI video workflows without requiring a mode enum field. The active mode is selected deterministically by the combination of conditioning inputs supplied:

| Workflow Mode | Conditioning Fields Present in `VideoTaskArgs` | Representative Diffusers Pipelines / Modular Blocks |
| --- | --- | --- |
| **1. Text-to-Video / Text-to-Audio-Video (`T2V` / `T2AV`)** | `prompt` only (all media fields empty/`null`) | `WanPipeline`, `HunyuanVideo15Pipeline`, `HunyuanVideoPipeline`, `LTX2Pipeline`, `LTX2ModularPipeline`, `MiniMaxH3ModularPipeline` (`t2va`), `CogVideoXPipeline`, `SkyReelsV2Pipeline`, `Kandinsky5T2VPipeline` |
| **2. Image-to-Video / Image-to-Audio-Video (`I2V` / `I2AV`)** | `prompt` + `image` | `WanImageToVideoPipeline` (including `Wan2.2-TI2V-5B` and `Wan2.2-I2V-A14B`), `HunyuanVideo15ImageToVideoPipeline`, `LTX2ImageToVideoPipeline`, `MiniMaxH3ModularPipeline` (`fl2va`), `CogVideoXImageToVideoPipeline`, `SkyReelsV2ImageToVideoPipeline`, `Kandinsky5I2VPipeline` |
| **3. First-Last-Frame-to-Video (`FLF2V` / `FLF2AV`)** | `prompt` + `image` + `last_image` (or `last_image` alone on pipelines that support end-anchored generation) | `WanImageToVideoPipeline` (`last_image`), `LTX2ConditionPipeline` / `LTX2ModularPipeline` (`index=0` and `index=-1`), `MiniMaxH3ModularPipeline` (`fl2va`) |
| **4. Arbitrary Keyframe-Anchored Generation** | `prompt` + `keyframes` (pinned images/clips at explicit frame indices with per-anchor strength) | `LTX2ConditionPipeline`, `LTXConditionPipeline`, `LTX2ModularPipeline` (`condition`) |
| **5. Multi-Modal Reference-to-Video (`R2V` / `Ref2VA`)** | `prompt` + `references` (ordered list of subject/style images, motion/camera reference videos, and/or voice/SFX reference audio clips) | `WanVACEPipeline` (`reference_images`), `MiniMaxH3ModularPipeline` (`ref2va`), `LTX2ModularPipeline` (`in_context`) |
| **6. Video-to-Video & Masked Video Inpainting / Control (`V2V` / `VACE`)** | `prompt` + `video` (+ optional `mask_video`, `strength`, `conditioning_scale`, `references`) | `WanVideoToVideoPipeline`, `WanVACEPipeline`, `CogVideoXVideoToVideoPipeline`, `SkyReelsV2DiffusionForcingVideoToVideoPipeline` |
| **7. Character Animation & Character Replacement (`Animate`)** | `image` + `pose_video` + `face_video` (+ `background_video` + `mask_video` for replacement; or `image` + `video` on `WanAnimate2ModularPipeline`) | `WanAnimatePipeline`, `WanAnimate2ModularPipeline` |
| **8. Audio-Conditioned Video / Lip-Sync (`A2V` / `S2V`)** | `prompt` + `image` (or `references`) + `audio` (or audio item in `references`) | `MiniMaxH3ModularPipeline` (`ref2va`), `LTX2ConditionPipeline` / `LTX2ModularPipeline` |

---

## 2. Diffusers Video Parameter Landscape

Inspecting the call signatures across `diffusers` (`v0.41+`) pipelines and modular blocks shows how common concepts map to checkpoint-specific parameter names:

| Semantic Control | `Wan 2.1 / 2.2` (`Wan*Pipeline`) | `LTX-2 / 2.3 / 2.5` (`LTX2*Pipeline` / `Modular`) | `HunyuanVideo` (`1.5` & `13B`) | `MiniMax H3` (`MiniMaxH3ModularPipeline`) | `CogVideoX 1.5` / `SkyReels V2` |
| --- | --- | --- | --- | --- | --- |
| **Spatial size** | `width`, `height` | `width`, `height` | `width`, `height` (optional on 1.5 I2V) | `width`, `height` | `width`, `height` |
| **Clip length (frames)** | `num_frames` ($4k+1$, default `81` or `121`); `segment_frame_length` on `WanAnimate` | `num_frames` ($8k+1$, or `None` w/ `duration_head`) | `num_frames` ($4k+1$, default `121` or `129`) | `num_frames` ($17k+5$, snapped up) | `num_frames` (default `49` / `81`) |
| **Frame rate** | Export-only (`16` for A14B, `24` for 5B); `fps` on `WanAnimate2` | `frame_rate: float = 24.0` (passed into DiT & export) | Export-only (`15` or `24`) | Fixed `24` fps internal & export | Export-only (`8` or `16`) |
| **Denoising steps & schedule** | `num_inference_steps`, `timesteps` (V2V) | `num_inference_steps`, `sigmas`, `timesteps` | `num_inference_steps`, `sigmas` | `num_inference_steps` | `num_inference_steps`, `timesteps` |
| **Primary visual guidance** | `guidance_scale` | `guidance_scale` | `guidance_scale` or `true_cfg_scale` (`13B`) or guider (`1.5`) | CFG-distilled (no guidance arg) | `guidance_scale`, `use_dynamic_cfg` |
| **Secondary / MoE guidance** | `guidance_scale_2` (low-noise `transformer_2` stage) | — | — | — | — |
| **Audio & Spatio-Temporal guidance** | — | `audio_guidance_scale`, `stg_scale`, `audio_stg_scale`, `modality_scale`, `audio_modality_scale`, `guidance_rescale`, `audio_guidance_rescale` | — | — | — |
| **Prompt token cap** | `max_sequence_length` (`512`) | `max_sequence_length` (`1024`) | `max_sequence_length` (`256`) | Handled by Qwen3-VL tokenizer | `max_sequence_length` (`226`) |
| **LoRA target modules** | `transformer`, `transformer_2` (`load_into_transformer_2`) | `transformer`, `connectors` | `transformer` | `transformer`, `transformer_ref` (`load_into_transformer_ref`) | `transformer` |
| **Output payload** | `.frames` (`np` / `pil`) | `(video, audio)` or `.frames` + `.audio` | `.frames` | `.videos` + `.audio` + `.sampling_rate` | `.frames` |

### Design Principle: Minimal Canonical Core + Passthrough `pipeline_args`

Following `ImageTaskArgs` and `GPTTaskArgs` (`template_args`):
1. **`VideoTaskConfig` retains only universal workload, geometry, and determinism parameters** required across all models and by Relay workload estimation: `width`, `height`, `num_frames`, `fps`, `steps`, `seed`, and the single universal `guidance_scale`. Each task produces exactly one video clip (`0.mp4`). The runtime adapts these to each pipeline's canonical parameter names (`fps` vs `frame_rate`, `guidance_scale` vs `true_cfg_scale`, `num_frames` vs `segment_frame_length`).
2. **All model-family-specific generation parameters are passed through a top-level `pipeline_args: dict[str, Any] | None = None` map** (directly analogous to `GPTTaskArgs.template_args`), without hardcoding model-specific fields in `VideoTaskArgs`. This covers `Wan 2.2`'s `guidance_scale_2`, `LTX-2.x`'s `audio_guidance_scale`, `stg_scale`, `audio_stg_scale`, `modality_scale`, `audio_modality_scale`, `guidance_rescale`, `sigmas`, `max_sequence_length`, `CogVideoX`'s `use_dynamic_cfg`, and future checkpoint kwargs without requiring schema migrations.
3. **Safety boundary on `pipeline_args`**: Entries MUST be JSON-serializable primitives/lists/dicts (no raw tensors or Python objects), MUST NOT shadow canonical `VideoTaskArgs` fields or internal execution-control parameters (`generator`, `latents`, `prompt_embeds`, `output_type`, `return_dict`, `callback_on_step_end`, `enable_prompt_enhancement`, `attention_kwargs`), and MUST match accepted parameters of the resolved `DiffusionPipeline.__call__` or `ModularPipeline` blocks/guiders. Unsupported keys fail fast with `TaskInvalid` before inference so callers never silently pay GPU time for ignored typos or wrong-model parameters.

---

## 3. Canonical `VideoTaskArgs` Specification

### 3.1 Shared Primitives Reused Unchanged from `image_task`

`VideoTaskArgs` MUST import and reuse `BaseModelArgs`, `ComponentSource`, `QuantizationConfig`, `FloatFractionAsInt`, and `NonEmptyString` directly so Relay model-file resolution, `ExecutionModelFileHashes`, GGUF/Diffusers component loading, and BitsAndBytes quantization work identically across image and video tasks.

### 3.2 Media Input Encoding Contract

All media inputs inside `VideoTaskArgs` MUST be self-contained **base64 data URLs** (no external HTTP URLs and no host filesystem paths, preserving offline worker execution and VSS parameter privacy):

- **Image data URL**: `data:image/<png|jpeg|webp>;base64,<payload>` — decoded to `PIL.Image.Image` in `RGB` mode.
- **Video data URL**: `data:video/<mp4|webm>;base64,<payload>` — decoded via `PyAV` (`av`) into an exact RGB frame list plus its container `fps` (and embedded audio waveform + `sample_rate` when present).
- **Audio data URL**: `data:audio/<wav|mp3|aac|flac|ogg>;base64,<payload>` — decoded via `PyAV` into a floating-point waveform tensor plus its `sample_rate`.

### 3.3 Implementation and JSON Schema References

The authoritative Pydantic data models, schema-level invariants, reserved passthrough key set (`RESERVED_PIPELINE_ARG_KEYS`), and canonical input-mode derivation (`derive_video_input_mode`) live directly in code and generated JSON Schema:

- Pydantic models and validators: [`../../sd_task/task_args/video_task/task_args.py`](../../sd_task/task_args/video_task/task_args.py)
- Package exports: [`../../sd_task/task_args/video_task/__init__.py`](../../sd_task/task_args/video_task/__init__.py) and [`../../sd_task/task_args/__init__.py`](../../sd_task/task_args/__init__.py)
- Generated JSON Schema: [`../../schema/video-task.json`](../../schema/video-task.json) (regenerated via [`../../generate_json_schema.py`](../../generate_json_schema.py))
- Validation unit tests: [`../../tests/test_video_task_args.py`](../../tests/test_video_task_args.py)

---

## 4. Field-by-Field Semantics and Validation Rules

### 4.1 Model, Pipeline & Modular Workflow Resolution, Dtype, Components, and Quantization

1. **`base_model` (`BaseModelArgs`)**:
   - Identifies the complete Hugging Face Diffusers pipeline repository containing either `model_index.json` (`DiffusionPipeline`) or `modular_model_index.json` (`ModularPipeline`).
2. **Automatic Modular Workflow Resolution & Unified `input_mode` Cache Identity**:
   - Clients do not pass a `workflow` field. Both Relay (`DeriveTaskModelRequirements`) and the worker runtime MUST derive the canonical `input_mode` purely from the presence of `VideoTaskArgs` conditioning fields using the exact same deterministic precedence:
     - `animate` is set $\to$ `"animate"`
     - `video` is set $\to$ `"v2v"`
     - `references` is non-empty $\to$ `"r2v"`
     - `keyframes` is non-empty $\to$ `"keyframes"`
     - `last_image` is set $\to$ `"flf2v"`
     - `audio` is set (with `image`) $\to$ `"a2v"`
     - `image` is set $\to$ `"i2v"`
     - otherwise (text-only `prompt`) $\to$ `"t2v"`
   - When the repository resolves to a `ModularPipeline` whose blocks define `_workflow_map` (such as `MiniMaxH3Blocks`, `LTX2AutoBlocks` / `LTX25AutoBlocks`, `HunyuanVideo15AutoBlocks`, or `WanImage2VideoAutoBlocks`), the runtime maps `input_mode` / active trigger inputs against `blocks._workflow_map` before calling `load_components()`. Passing the resolved `workflow` into `ModularPipeline.from_pretrained(..., workflow=resolved_workflow)` prunes unused checkpoint partitions (for example loading only `transformer/` for `t2va`/`fl2va` vs. `transformer_ref/` for `ref2va` on `MiniMax-H3`). If no workflow in `_workflow_map` matches the supplied conditioning combination, resolution MUST fail immediately with `TaskInvalid` before loading weights.
   - Because Relay uses `RequestedModelCacheHash` to compute node `localityWeight` (`+0.3`), set `model_switched` for execution `Timeout` calculation, and aggregate `loaded_models` (`min_vram` and `in_memory_node_count`), **both Relay's `RequestedModelCacheConfig` / `RequestedModelCacheHash` and the worker-local `ModelCache` key MUST include `input_mode`**. Relay derives `input_mode` directly from `TaskArgs` without querying Hugging Face, ensuring 1-to-1 alignment between Relay's in-memory locality/switch decisions and actual GPU-resident pipeline/partition state on the node.
3. **`dtype`**:
   - Resolved before weight loading to one concrete floating compute dtype (`float16`, `bfloat16`, or `float32`) using the same rule as `image_task`: when `"auto"`, inspect the active denoiser (`transformer`, `transformer_2`, or `transformer_ref`) `config.json` `torch_dtype`, falling back to `float32` if absent/`null`/`"auto"`.
4. **`components` & `quantization_config`**:
   - Identical container contract to `image_task` (`../quantization.md`), extended to recognize multi-denoiser video repositories:
     - `components` supports `"diffusers"` (any component in `model_index.json` / `modular_model_index.json`, including alternate `text_encoder` replacements for `LTX-2.x` / `MiniMax-H3`, standalone `vae`, or quantized transformers) and `"gguf"` (supporting `transformer`, `transformer_2` in `Wan 2.2`, and `transformer_ref` in `MiniMax-H3`).
     - `quantization_config.components_to_quantize` supports `["transformer"]`, `["transformer", "transformer_2"]`, `["transformer", "text_encoder"]`, etc.

### 4.2 Prompt and Conditioning Inputs

1. **`prompt` & `negative_prompt`**:
   - At least one of `prompt`, `image`, `video`, or `animate` MUST be non-empty; pure text-to-video requires a non-empty `prompt`.
   - `negative_prompt` follows `image_task` validation: a non-empty `negative_prompt` MUST be rejected with `TaskInvalid` when the resolved pipeline has no `negative_prompt` parameter (such as guidance-distilled `MiniMaxH3ModularPipeline`) or when its effective CFG scale is $\le 1.0$ without negative guidance active.
   - Built-in LLM prompt rewriting inside pipelines (such as `LTX2Pipeline`'s `enable_prompt_enhancement`) MUST be forced to `False` so prompt rendering stays deterministic and faithful to `TaskArgs`.
2. **`image`, `last_image`, and `keyframes` (Mutually Coordinated Frame Anchoring)**:
   - `image` and `last_image` are ergonomic top-level fields for the two most common short-drama modes (`I2V` and `FLF2V`).
   - `keyframes` generalizes frame pinning to $1 \dots N$ frames at arbitrary 0-based pixel frame indices (`frame_index`, where `-1` denotes `num_frames - 1`).
   - Validation rules:
     - `keyframes` MUST NOT be combined with `image` or `last_image` in the same request (use either `image`/`last_image` or `keyframes`, never both).
     - When `image` + `last_image` are provided to a pipeline that exposes `conditions` (`LTX2ConditionPipeline` / `LTX2ModularPipeline`) instead of `(image, last_image)`, the input adapter MUST automatically translate `image` to `index=0, strength=1.0` and `last_image` to `index=-1, strength=1.0`.
     - Conversely, if `last_image` or `keyframes` is supplied to an I2V-only pipeline (such as `HunyuanVideo15ImageToVideoPipeline` or `CogVideoXImageToVideoPipeline`) that accepts neither `last_image` nor `conditions`, the runtime MUST fail fast with `TaskInvalid` before running denoising.
3. **`references` (`list[MediaReference]`)**:
   - Ordered list of multimodal reference items (`type` $\in$ `{"image", "video", "audio"}`). Order MUST be preserved verbatim because `MiniMax-H3` binds `<Picture i>`, `<Video j>`, and `<Audio k>` prompt tags and 3D RoPE positions strictly by list order.
   - Adapter mapping:
     - On `WanVACEPipeline`: requires all items in `references` to have `type == "image"` and maps them to `reference_images: list[PIL.Image]`. Any `"video"` or `"audio"` reference on `WanVACEPipeline` raises `TaskInvalid`.
     - On `MiniMaxH3ModularPipeline`: converts `"image"` $\to$ `MiniMaxH3ImageReference`, `"video"` $\to$ `MiniMaxH3VideoReference` (preserving decoded `fps` and embedded soundtrack), and `"audio"` $\to$ `MiniMaxH3AudioReference` (preserving `sample_rate`), enforcing H3's limits ($\le 9$ images, $\le 3$ videos, $\le 3$ audios, $\le 12$ total, not audio-only).
     - On single-reference I2V pipelines that accept `image: list[PIL.Image]` without `strength`, image-only `references` MAY map if `image` is unset; otherwise unsupported `references` raise `TaskInvalid`.
4. **`video`, `mask_video`, `strength`, `conditioning_scale`, `audio`, and `animate`**:
   - `video`: Input driving or source video clip for V2V (`WanVideoToVideoPipeline`, `CogVideoXVideoToVideoPipeline`), controllable generation (`WanVACEPipeline`), or motion driving (`WanAnimate2ModularPipeline` mapped to `driving_video` + `driving_video_fps`).
   - `mask_video`: Grayscale/RGB mask video for `WanVACEPipeline` (`mask`) or `WanAnimatePipeline` (`mask_video`). Requires `video` or `animate` to be present.
   - `strength`: Denoising strength $\in [0, 1]$ forwarded only to V2V pipelines exposing `strength`.
   - `conditioning_scale`: Control stream weight forwarded only to controllable pipelines exposing `conditioning_scale` (such as `WanVACEPipeline`).
   - `audio`: Standalone conditioning audio track for audio-to-video / lip-sync pipelines; on `MiniMaxH3ModularPipeline` when `audio` is provided alongside `image` without `references`, or on audio-driven pipelines, mapped to the pipeline's audio conditioning input.
   - `animate`: Structured container for character animation/replacement pipelines (`WanAnimatePipeline` requiring `pose_video` + `face_video`, and `background_video` + `mask_video` when `mode == "replace"`).

### 4.3 `task_config` (`VideoTaskConfig`)

1. **`width` & `height` (`int | None`)**:
   - Optional so that I2V / FLF2V / Ref2VA checkpoints with aspect-ratio auto-sizing (such as `HunyuanVideo15ImageToVideoPipeline`, `MiniMaxH3ModularPipeline`, and `WanAnimate2ModularPipeline`) can derive canvas dimensions from the input `image` when omitted, while T2V tasks fall back to checkpoint defaults if unset (`None`) or use explicit client dimensions.
   - **Strict Spatial Divisibility Check (No Silent Cropping/Rounding)**: Notice that `WanPipeline` logs a warning and silently rounds `height`/`width` down if they are not divisible by `vae_scale_factor_spatial * patch_size`. Silent dimension rounding changes workload size and breaks predictability. Therefore, when explicit `width` or `height` is provided, `stable-diffusion-task` MUST compute the pipeline's required spatial multiple $M_s$ (e.g., `16` for Wan/Hunyuan/Kandinsky, `32` for LTX-2/MiniMax-H3, `8` for CogVideoX) and reject non-multiples with `TaskInvalid` before inference!
2. **`num_frames` (`int = 81`)**:
   - Explicit frame count (never left to dynamic `duration_head` prediction in Crynux tasks, because Relay requires a deterministic workload estimate `VideoUnits` at task creation time and VSS requires every validation node to render the exact same number of frames).
   - **Strict Temporal Grid Validation**: Before denoising, the runtime MUST verify that `num_frames` satisfies the loaded checkpoint's temporal VAE grid—e.g., `(num_frames - 1) % 4 == 0` for $4\times$ causal VAEs (`Wan`, `HunyuanVideo`), `(num_frames - 1) % 8 == 0` for $8\times$ causal VAEs (`LTX-2`), or `num_frames % 17 == 5` ($5\text{s} \le T/24 \le 15\text{s}$) for `MiniMax-H3`—raising `TaskInvalid` rather than silently mutating `num_frames`! (For `WanAnimatePipeline` / `WanAnimate2`, `num_frames` maps to segment/clip target length or validates driving video length.)
3. **`fps` (`float | None`)**:
   - If the pipeline call signature accepts `frame_rate` (`LTX2*`) or `fps` (`StableVideoDiffusionPipeline`, `WanAnimate2ModularPipeline`), the resolved `fps` MUST be forwarded into the model call AND used when muxing `0.mp4`.
   - If omitted (`None`), the runtime resolves `fps` from the checkpoint's native frame rate property/default (e.g., `24.0` on `LTX-2` / `MiniMax-H3` / `Wan 2.2 5B`, `16.0` on `Wan 2.1 / 2.2 A14B` / `CogVideoX 1.5`, `15.0` or `24.0` on `HunyuanVideo 1.5`).
4. **`steps` & `seed`**:
   - `steps` maps to `num_inference_steps`.
   - `seed` seeds Python `random`, `numpy.random`, `torch.manual_seed`, and constructs `torch.Generator(device="cpu").manual_seed(seed)`.
   - Every task generates a single video clip (`num_videos_per_prompt` is fixed to `1` when supported by the pipeline signature and prohibited in `pipeline_args`).
5. **Guidance (`guidance_scale`)**:
   - `task_config.guidance_scale` is the single canonical visual guidance field, mapping through the same capability-detection pattern as `image_task/guidance.py` (`guidance_scale` vs. `true_cfg_scale`, or `guider` configuration on `ModularPipeline`).

### 4.4 Passthrough `pipeline_args` (`dict[str, Any] | None`)

Model-family-specific generation controls MUST NOT be hardcoded onto `VideoTaskConfig`. Instead, `VideoTaskArgs.pipeline_args` serves as an optional raw passthrough extension map (following the `GPTTaskArgs.template_args` pattern):

1. **Covered Model-Specific Parameters**:
   - Dual-expert MoE guidance (`Wan 2.2` `guidance_scale_2`);
   - Joint audio-video & spatio-temporal guidance (`LTX-2.x` `audio_guidance_scale`, `stg_scale`, `audio_stg_scale`, `modality_scale`, `audio_modality_scale`, `guidance_rescale`, `audio_guidance_rescale`, `spatio_temporal_guidance_blocks`, `noise_scale`, `decode_timestep`, `decode_noise_scale`, `use_cross_timestep`);
   - Scheduler & text-encoder overrides (`sigmas`, `timesteps`, `eta`, `max_sequence_length`, `use_dynamic_cfg`); and
   - Future pipeline-specific scalar/list options added by upstream `diffusers`.
2. **Routing to `DiffusionPipeline` vs. `ModularPipeline`**:
   - On a `DiffusionPipeline`, compatible keys in `pipeline_args` are forwarded directly as keyword arguments to `pipeline.__call__(...)`.
   - On a `ModularPipeline` (where guidance parameters such as `guidance_scale`, `guidance_scale_2`, `audio_guidance_scale`, `stg_scale`, and `modality_scale` live on `pipeline.guider` / `pipeline.guider_2` / `pipeline.audio_guider` configs rather than block inputs), the runtime inspects both the block `inputs` and the pipeline's registered guider config fields (`guider`, `guider_2`, `audio_guider`), cloning/updating the per-call guider state without mutating the cached pipeline instance across tasks, and forwards remaining block-input keys to `pipeline(...)`.
3. **Reserved Keys & Validation**:
   - `pipeline_args` MUST contain only JSON-compatible primitives, lists, or dicts.
   - `pipeline_args` MUST NOT contain canonical `VideoTaskArgs` fields or internal runtime/nondeterministic hooks; any key in `RESERVED_PIPELINE_ARG_KEYS` (defined in [`../../sd_task/task_args/video_task/task_args.py`](../../sd_task/task_args/video_task/task_args.py)) MUST be rejected during `VideoTaskArgs` validation.
   - Any key in `pipeline_args` that is accepted by neither the resolved pipeline call signature nor its active modular block inputs / guider configs MUST fail before generation with `TaskInvalid`.

### 4.5 Multi-LoRA Stack (`loras: list[VideoLoraArgs]`)

Video workflows routinely stack 2 to 4 LoRAs simultaneously (for example: a `LightX2V` 4-step speed LoRA + a Character LoRA + a Camera/Motion LoRA, or separate high-noise and low-noise expert LoRAs on `Wan 2.2`).

1. Each entry in `loras` is loaded sequentially in list order with deterministic adapter names `["lora_0", "lora_1", ...]` and activated without weight fusion (`set_adapters` with per-adapter weights).
2. **`revision` Pinning**: Unlike legacy single-LoRA `LoraArgs`, `VideoLoraArgs` includes `revision: str | None = None` so Relay can commit-pin LoRA repositories in `ExecutionModelFileHashes` alongside `base_model` and `components`.
3. **`target` Routing for Dual-Denoiser Architectures**:
   - `"auto"` (default): Loads into primary `transformer` (or `transformer_ref` when `MiniMaxH3ModularPipeline` is running `workflow="ref2va"`).
   - `"transformer"`: Explicitly loads into `pipeline.transformer`.
   - `"transformer_2"`: Passes `load_into_transformer_2=True` (`Wan 2.2` low-noise expert); fails with `TaskInvalid` if `pipeline` has no `transformer_2`.
   - `"transformer_ref"`: Passes `load_into_transformer_ref=True` (`MiniMax-H3` reference partition); fails with `TaskInvalid` if `pipeline` has no `transformer_ref`.
   - `"both"`: Loads the adapter into both `transformer` and `transformer_2` with the same weight when `transformer_2` exists; fails with `TaskInvalid` otherwise.
4. The complete ordered `loras` list (including resolved `revision`, `weight_file_name`, `weight`, and `target`) MUST participate in `ModelCache` key identity.

### 4.6 Deterministic Output Boundary (`VideoTaskResult`)

Following the same raw-output principle as `ImageTaskArgs` (which always writes lossless PNGs without client-configurable encoding options) and `GPTTaskArgs` (which always returns raw decoded text):

1. Pipeline execution requests unencoded RGB frames (`output_type="np"` or `"pil"`) plus any generated audio tensor (`audio` and `audio_sampling_rate`).
2. Whenever the resolved pipeline produces an audio waveform, it MUST be muxed as an AAC stereo stream (`aac`, checkpoint native sample rate) into `0.mp4`; silent video pipelines write a video-only `0.mp4`. Any audio stripping, transcoding, or re-compression belongs to downstream consumers outside `stable-diffusion-task`.
3. Encoding MUST use fixed canonical `PyAV` settings (`libx264`, `pix_fmt="yuv420p"`, constant frame rate `resolved_fps`, fixed runtime `crf=18`, and deterministic single-thread x264 options `{"threads": "1"}`) writing a single `0.mp4` artifact (`TaskSize = 1`).
4. `run_video_task` returns `VideoTaskResult`, exposing:
   - the output path or encoded video bytes of the generated clip;
   - `num_frames`, `width`, `height`, `fps`, and `has_audio`;
   - `execution_dtype` (resolved from `transformer`, `transformer_2`, or `transformer_ref`); and
   - `execution_quantization`.

---

## 5. Relay Workload & Model-File Integration Summary

Placing `video_task` inside `stable-diffusion-task` keeps node/worker integration minimal while cleanly separating image vs. video scheduling on Relay:

1. **Model File Extraction & `RequestedModelCacheConfig` (Extending `ImageTaskArgs`)**:
   - `base_model` $\to$ `(repo=base_model.name, format="diffusers", revision=base_model.revision)`
   - Each `components[k]` $\to$ `(repo=source.repo, format=source.format, revision=source.revision, format_kwargs=source.format_kwargs)`
   - Each `loras[i]` $\to$ `(repo=lora.model, format="lora", revision=lora.revision, format_kwargs={"weight_file_name": lora.weight_file_name})`
   - `RequestedModelCacheConfig` hashes `{"task_kind": "video", "input_mode": <derived>, "dtype": <normalized>, "base_model": ..., "components": ..., "loras": ..., "quantization_config": ...}` (excluding revisions).
2. **Relay Workload Formula (`VideoUnits`)**:
   - `TaskSize` is fixed to `1` (matching `TaskTypeLLM`), and Relay computes:
     ```text
     video_units = resolved_width * resolved_height * num_frames * steps
     ```
     (When `width`/`height` are omitted in I2V, either require explicit `width`/`height` at Relay task creation or derive dimensions from the decoded base64 `image` header + checkpoint default canvas rule at create time, just as Relay already decodes VLM base64 image dimensions in `task_execution_parameters.md`.)

---

## 6. Canonical JSON Examples Across Core Scenarios

### Example 1: Short-Drama I2V with Wan 2.2 A14B + Dual-Expert 4-Step Speed LoRA + Character LoRA (24 GB Quantized)

```json
{
  "version": "3.8.0",
  "base_model": {
    "name": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    "revision": null
  },
  "dtype": "bfloat16",
  "prompt": "Medium vertical shot, the young woman turns her head toward the window as rain streaks down the glass, subtle wind moving her hair, cinematic moody lighting, smooth dolly-in",
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "task_config": {
    "width": 720,
    "height": 1280,
    "num_frames": 81,
    "fps": 16,
    "steps": 4,
    "seed": 42,
    "guidance_scale": 1.0
  },
  "pipeline_args": {
    "guidance_scale_2": 1.0
  },
  "loras": [
    {
      "model": "lightx2v/Wan2.2-Lightning",
      "weight_file_name": "wan2.2_i2v_A14b_high_noise_lora_rank64_lightx2v_4step.safetensors",
      "weight": 100,
      "target": "transformer"
    },
    {
      "model": "lightx2v/Wan2.2-Lightning",
      "weight_file_name": "wan2.2_i2v_A14b_low_noise_lora_rank64_lightx2v_4step.safetensors",
      "weight": 100,
      "target": "transformer_2"
    },
    {
      "model": "studio-assets/protagonist-character-wan22-lora",
      "weight_file_name": "heroine_v3.safetensors",
      "weight": 85,
      "target": "both"
    }
  ],
  "quantization_config": {
    "quant_method": "bitsandbytes",
    "components_to_quantize": ["transformer", "transformer_2"],
    "quant_kwargs": {
      "load_in_4bit": true,
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_compute_dtype": "bfloat16",
      "bnb_4bit_use_double_quant": false
    }
  }
}
```

### Example 2: Joint Audio-Video Dialogue Shot with LTX-2.5 + Custom Text Encoder Override + First-Last-Frame Pinning

```json
{
  "version": "3.8.0",
  "base_model": {
    "name": "Lightricks/LTX-2.5",
    "revision": null
  },
  "dtype": "bfloat16",
  "prompt": "Close-up shot in a dimly lit bar. She leans forward and whispers in English: \"We only get one chance tonight.\" Soft jazz plays faintly in the background.",
  "negative_prompt": "shaky, glitchy, low quality, deformed hands, distorted face, static",
  "image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "last_image": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA...",
  "task_config": {
    "width": 704,
    "height": 1280,
    "num_frames": 121,
    "fps": 24,
    "steps": 8,
    "seed": 1024,
    "guidance_scale": 3.0
  },
  "pipeline_args": {
    "audio_guidance_scale": 7.0,
    "stg_scale": 1.0,
    "modality_scale": 3.0
  },
  "components": {
    "text_encoder": {
      "format": "diffusers",
      "repo": "org/gemma-4-12b-ltx25-text-encoder",
      "revision": null,
      "format_kwargs": {
        "subfolder": "text_encoder"
      }
    }
  }
}
```

### Example 3: Multi-Reference Audio-Video Generation (`Ref2VA`) with MiniMax H3 Modular Pipeline

```json
{
  "version": "3.8.0",
  "base_model": {
    "name": "MiniMaxAI/MiniMax-H3",
    "revision": null
  },
  "dtype": "bfloat16",
  "prompt": "Character from <Picture 1> stands inside the hall from <Picture 2> and delivers the line with the voice timbre of <Audio 1>.",
  "references": [
    {
      "type": "image",
      "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "type": "image",
      "data": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."
    },
    {
      "type": "audio",
      "data": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAA..."
    }
  ],
  "task_config": {
    "width": 768,
    "height": 1344,
    "num_frames": 124,
    "fps": 24,
    "steps": 8,
    "seed": 77
  },
  "loras": [
    {
      "model": "lightx2v/Minimax-h3-Turbo",
      "weight_file_name": "minimax_h3_8step_lora.safetensors",
      "weight": 100,
      "target": "transformer_ref"
    }
  ]
}
```
