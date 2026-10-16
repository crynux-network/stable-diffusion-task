# Image Quantization

## Scope and authority

This document specifies quantized image execution from canonical
`ImageTaskArgs` to generated images. It extends the common execution contract
in [model-compatibility.md](model-compatibility.md).

The TaskArgs schema MUST provide stable backend-neutral containers. Current
backend names, keys, values, components, and combinations MUST NOT be encoded
as a closed TaskArgs enum or an upstream execution allowlist.
stable-diffusion-task MUST validate a configuration against its runtime
adapter, official configuration class, installed libraries, hardware,
pipeline metadata, and component metadata.

The configurations in this document define current runtime adapter coverage.
They MUST NOT be interpreted as model-ID allowlists.

## TaskArgs contract

### Base model

`base_model` MUST always identify the complete Diffusers pipeline source:

```json
{
  "name": "org/pipeline",
  "revision": null
}
```

`name` MUST identify a complete Diffusers pipeline repository containing
`model_index.json`. Nunchaku Lite MUST use this form. A standalone Nunchaku
transformer safetensors file is not a complete pipeline and MUST be
rejected.

An embedded quantized component declared by the complete pipeline MUST NOT be
repeated in TaskArgs.

### Independent components

`components` MUST be absent or an object that maps a component name from
`model_index.json` to an independent model-file source.

An independent GGUF transformer uses:

```json
{
  "base_model": {
    "name": "org/pipeline",
    "revision": null
  },
  "components": {
    "transformer": {
      "format": "gguf",
      "repo": "org/quantized-transformer",
      "revision": null,
      "format_kwargs": {
        "filename": "transformer-Q4_K_M.gguf"
      }
    }
  }
}
```

Each component value MUST use this stable shape:

```json
{
  "format": "gguf",
  "repo": "org/quantized-transformer",
  "revision": null,
  "format_kwargs": {
    "filename": "transformer-Q4_K_M.gguf"
  }
}
```

`format` and `repo` MUST be non-empty strings. `revision` MUST be absent,
`null`, or a repository ref. `format_kwargs` MUST be absent or a JSON object.
The TaskArgs schema MUST NOT enumerate component names, file formats, or
format-specific keys.

The current GGUF adapter MUST support a `transformer` component with
`format="gguf"` and MUST require `format_kwargs.filename`. The generic
component container MUST remain capable of representing other formats and
components for additional adapters.

The loader MUST read `model_index.json`, resolve the declared transformer
class, and invoke that model class through `from_single_file`. The loader MUST
resolve the transformer Diffusers config from the base pipeline repository
component subfolder through the configured Hugging Face cache directory, and
MUST pass that local config directory to `from_single_file`. The loader MUST pass Diffusers `GGUFQuantizationConfig` with `compute_dtype` equal to the
task `dtype`, or `float32` when `dtype` is absent. After `from_single_file`,
the loader MUST materialize every Diffusers `GGUFParameter` whose GGUF type is
an unquantized floating-point type (`F32`, `F16`, or `BF16`) into a plain
`torch.nn.Parameter` at `compute_dtype` through Diffusers
`dequantize_gguf_tensor`. Quantized GGUFParameter tensors such as `Q4_K` MUST
remain GGUF-backed. The loaded GGUF transformer MUST be injected before the
remaining pipeline components are loaded. The loader MUST NOT download or load
the overridden floating-point transformer from the base repository. It MUST
NOT call `DiffusionPipeline.from_single_file` and MUST NOT choose a class from
a model ID or filename.

An independent Diffusers-format component uses:

```json
{
  "base_model": {
    "name": "org/pipeline",
    "revision": null
  },
  "components": {
    "vae": {
      "format": "diffusers",
      "repo": "org/standalone-vae",
      "revision": null,
      "format_kwargs": {}
    }
  }
}
```

The current Diffusers component adapter MUST support any component name
declared in the base pipeline `model_index.json` with `format="diffusers"`.
`format_kwargs.subfolder` MUST be absent, or a non-empty string naming the
subdirectory inside the component repository. The adapter MUST NOT require
`filename`.

The Diffusers component loader MUST read `model_index.json`, resolve the
declared component class for that component name, and invoke that class through
`from_pretrained` on the prepared component path. When `subfolder` is set, the
loader MUST pass it to `from_pretrained`. The loader MUST inject the loaded
module into `pipeline_class.from_pretrained` under that component name. The
loader MUST NOT download or load the overridden component from the base
repository. It MUST NOT choose a class from a model ID or filename.

Every requested repository ref MUST be resolved to an immutable commit before
execution. The pipeline commit, component name, component format, component
model-file commit, and format kwargs that participate in download identity
MUST participate in model-file identity. For GGUF components that includes
`filename`. For Diffusers components that includes `subfolder` when present.

### Task-time quantization

`quantization_config` MUST be absent, `null`, or an object with this stable
shape:

```json
{
  "quant_method": "bitsandbytes",
  "components_to_quantize": ["transformer"],
  "quant_kwargs": {}
}
```

`quant_method` MUST be a non-empty string and MUST use the same method name as
the corresponding gpt-task configuration.
`components_to_quantize` MUST be a non-empty array of unique component names.
`quant_kwargs` MUST be a JSON object.

The TaskArgs schema MUST NOT enumerate backend names, component names,
backend-specific keys, or backend-specific values.

stable-diffusion-task MUST select an adapter by `quant_method`. An adapter MUST
construct the corresponding official Diffusers or Transformers configuration
explicitly and MUST NOT forward unchecked keyword arguments to a third-party
constructor.

The complete canonical `quantization_config` MUST participate in execution
identity.

An existing component with embedded quantization and a task-time
configuration targeting the same component MUST be rejected. Embedded and
task-time configurations MUST NOT be merged or overridden.

## Common BitsAndBytes configurations

### 4-bit

The current adapter MUST support:

```json
{
  "quant_method": "bitsandbytes",
  "components_to_quantize": ["transformer"],
  "quant_kwargs": {
    "load_in_4bit": true,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": false
  }
}
```

The adapter MUST support:

- `bnb_4bit_quant_type` values `nf4` and `fp4`;
- `bnb_4bit_compute_dtype` values `float16`, `bfloat16`, and `float32`; and
- both boolean values for `bnb_4bit_use_double_quant`.

When `bnb_4bit_compute_dtype` is absent, the runtime MUST resolve the task
`dtype`. The resolved value MUST participate in execution and cache identity.

### INT8

The current adapter MUST support:

```json
{
  "quant_method": "bitsandbytes",
  "components_to_quantize": ["transformer"],
  "quant_kwargs": {
    "load_in_8bit": true,
    "llm_int8_threshold": 6.0,
    "llm_int8_skip_modules": null
  }
}
```

The adapter MUST pass `llm_int8_threshold` through the official configuration
class. Tests MUST cover the common values `0.0` and `6.0`.

`llm_int8_skip_modules` MUST remain representable. Its effective value MUST
participate in execution and cache identity.

CPU and disk offload MUST be prohibited. `llm_int8_has_fp16_weight` MUST be
false for network inference.

### Component selection

The runtime MUST support quantizing one compatible component or multiple
compatible components with one backend configuration. Current coverage MUST
include:

- a Diffusers transformer by itself; and
- a Diffusers transformer together with a Transformers text encoder.

Each component name MUST exist in `model_index.json`. The resolved component
class MUST support the selected official BitsAndBytes configuration class.
Class compatibility MUST be derived from metadata and installed registries, not
from a model-ID allowlist.

One task-time configuration selects one dynamic backend and one kwargs object
for every listed component. Per-component `quant_mapping` is outside the
current TaskArgs contract.

## Prequantized model execution

### Nunchaku Lite INT4

Nunchaku Lite MUST be loaded only from a complete Diffusers pipeline
repository whose component configuration selects the Nunchaku runtime.
TaskArgs MUST NOT duplicate the embedded backend configuration.

Nunchaku Lite execution requires NVIDIA CUDA kernels and is therefore limited
to Linux CUDA hosts in practice. Windows MUST return an unsupported-runtime
error when the Lite kernel package is unavailable. macOS MUST NOT be rejected
by an sd-task platform gate; loading MUST fail through the missing CUDA kernel
or dependency error.

The verified Linux baseline used a 24 GB RTX 4090. Peak allocated CUDA memory
was approximately 6.8 GiB for 512 by 512 generation and 8.6 GiB for 1024 by
1024 generation with nine steps.

Three repeated 1024 by 1024 executions produced perceptual-hash distances of
6. Cross-node tests MUST record their distance distribution against the
validation pool's configured image distance threshold. These test results
MUST NOT form a model or configuration execution allowlist.

### Component GGUF

GGUF execution MUST retain quantized weights in low-memory storage and
dynamically dequantize them to the selected compute dtype during execution.
A path that expands the complete model file to persistent floating-point
weights MUST NOT be reported as GGUF quantized execution.

The loader MUST recognize the Diffusers-supported tensor types:

- `BF16`;
- `Q4_0` and `Q4_1`;
- `Q5_0` and `Q5_1`;
- `Q8_0`; and
- `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, and `Q6_K`.

A mixed GGUF file MUST be validated from each tensor's actual type. The
runtime MUST NOT infer support from the filename.

Current verification coverage MUST include real Q4_0, Q4_K-family,
Q5_K-family, and Q8_0 model files.

The GGUF transformer MAY be combined with task-time BitsAndBytes quantization
of other unquantized compatible components. A component MUST NOT receive two
quantization methods.

## LoRA compatibility

The load order MUST be:

1. load the base pipeline and quantized components;
2. load floating-point LoRA weights; and
3. activate the adapter without fusing it into the base weights.

LoRA tests MUST cover:

- an unquantized base pipeline;
- BitsAndBytes 4-bit components; or
- BitsAndBytes INT8 components.

LoRA with Nunchaku Lite, GGUF, fine-grained FP8, AWQ, or GPTQModel components
MUST use the same load order when the underlying component API supports
adapter injection. The runtime MUST attempt the combination without consulting
a preverified combination list. An API or kernel incompatibility MUST
propagate through the existing execution-error contract.

The existing LoRA TaskArgs structure remains unchanged.

## Platform restrictions

### Windows CUDA

The current Windows CUDA dependency set supports:

- BitsAndBytes NF4 and FP4 4-bit component execution;
- BitsAndBytes INT8 component execution; and
- component GGUF execution after the required Diffusers and GGUF runtime probe
  succeeds.

Nunchaku Lite MUST return an unsupported-runtime error on Windows.

### Linux CUDA

The current Linux CUDA dependency set supports:

- BitsAndBytes NF4 and FP4 4-bit component execution;
- BitsAndBytes INT8 component execution;
- component GGUF execution; and
- Nunchaku Lite INT4 execution when its kernel is available.

### macOS

The current macOS dependency set includes BitsAndBytes and GGUF packages.
sd-task MUST NOT reject BitsAndBytes or GGUF solely because `sys.platform` is
`darwin`. Execution success depends on the installed wheels, Diffusers MPS
support, and Apple Silicon runtime behavior.

Nunchaku Lite has no MPS backend. macOS MUST NOT apply an sd-task platform
gate for Nunchaku; the load MUST fail through missing CUDA kernels or
dependencies.

Actual execution MUST satisfy the selected backend's dependency,
component-class, kernel, and GPU-architecture requirements. Availability of a
Python class alone MUST NOT be treated as proof that a task can execute.

## Runtime and cache requirements

Quantized image execution MUST use the existing serial pipeline execution
path. CPU and disk offload MUST remain disabled.

The model cache identity MUST include:

- the resolved pipeline class;
- every immutable model-file identity;
- the task dtype and every resolved component compute dtype;
- the complete task-time and embedded quantization configuration;
- the quantized component set;
- the selected backend and kernel-relevant configuration; and
- the complete LoRA configuration.

Cache entries with different effective configurations MUST NOT be shared.

The runtime MUST report the actual execution dtype, normalized effective
quantization set, component-to-backend mapping, and loaded model-file identity.
Execution dtype MUST be the floating compute or cast dtype
(`float16`, `bfloat16`, or `float32`). It MUST NOT report quantized packed
storage dtypes such as `uint8` or `int8`. When BitsAndBytes supplies
`bnb_4bit_compute_dtype`, that value MUST be reported. When TaskArgs `dtype`
is `auto`, the runtime MUST report the resolved concrete floating dtype, not
the literal `auto`. It MUST not report a requested backend when loading fell
back to floating-point weights.

## Unsupported execution

The current runtime adapters MUST reject:

- a task-time configuration that overlaps embedded component quantization;
- CPU or disk offload;
- more than one task-time dynamic backend;
- a standalone Nunchaku transformer safetensors file;
- a pipeline-level GGUF load that does not resolve the component class from
  `model_index.json`;
- fused LoRA on quantized weights; and
- a backend for which no runtime adapter exists.

The stable TaskArgs container remains capable of representing future
backends. Adding a backend MUST require only its runtime adapter, dependency
contract, cache identity, and tests; it MUST NOT require a TaskArgs schema
change or an upstream execution allowlist.

## Example TaskArgs for 24 GB CUDA

The following TaskArgs are coverage samples for common Hub checkpoints that
fit a single 24 GB CUDA device under the current adapters. They MUST NOT be
interpreted or implemented as a runtime model-ID allowlist.

### Task-time BitsAndBytes NF4 on FLUX.2 klein 4B

```json
{
  "version": "3.7.0",
  "base_model": {"name": "black-forest-labs/FLUX.2-klein-4B", "revision": null},
  "dtype": "bfloat16",
  "prompt": "A product photo of a white ceramic cup on a wooden table",
  "task_config": {
    "image_width": 1024,
    "image_height": 1024,
    "steps": 4,
    "seed": 0,
    "num_images_per_prompt": 1,
    "guidance_scale": 1
  },
  "components": {},
  "quantization_config": {
    "quant_method": "bitsandbytes",
    "components_to_quantize": ["transformer"],
    "quant_kwargs": {
      "load_in_4bit": true,
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_compute_dtype": "bfloat16",
      "bnb_4bit_use_double_quant": false
    }
  }
}
```

### Task-time BitsAndBytes NF4 on Z-Image Turbo

```json
{
  "version": "3.7.0",
  "base_model": {"name": "Tongyi-MAI/Z-Image-Turbo", "revision": null},
  "dtype": "bfloat16",
  "prompt": "A product photo of a white ceramic cup on a wooden table",
  "task_config": {
    "image_width": 1024,
    "image_height": 1024,
    "steps": 9,
    "seed": 0,
    "num_images_per_prompt": 1
  },
  "components": {},
  "quantization_config": {
    "quant_method": "bitsandbytes",
    "components_to_quantize": ["transformer"],
    "quant_kwargs": {
      "load_in_4bit": true,
      "bnb_4bit_quant_type": "nf4",
      "bnb_4bit_compute_dtype": "bfloat16"
    }
  }
}
```

### Component GGUF Q4_K_M transformer for Z-Image Turbo

```json
{
  "version": "3.7.0",
  "base_model": {"name": "Tongyi-MAI/Z-Image-Turbo", "revision": null},
  "dtype": "bfloat16",
  "prompt": "A product photo of a white ceramic cup on a wooden table",
  "task_config": {
    "image_width": 1024,
    "image_height": 1024,
    "steps": 9,
    "seed": 0,
    "num_images_per_prompt": 1
  },
  "components": {
    "transformer": {
      "format": "gguf",
      "repo": "jayn7/Z-Image-Turbo-GGUF",
      "revision": null,
      "format_kwargs": {
        "filename": "z_image_turbo-Q4_K_M.gguf"
      }
    }
  },
  "quantization_config": null
}
```

### Nunchaku Lite INT4 Z-Image Turbo

Linux CUDA only. The base model MUST be a complete Diffusers pipeline
repository that already declares the Nunchaku component runtime. TaskArgs
MUST NOT duplicate the embedded backend configuration.

```json
{
  "version": "3.7.0",
  "base_model": {
    "name": "lite-infer/z-image-turbo-nunchaku-lite-int4_r32-bnb4-text-encoder",
    "revision": null
  },
  "dtype": "bfloat16",
  "prompt": "A product photo of a white ceramic cup on a wooden table",
  "task_config": {
    "image_width": 1024,
    "image_height": 1024,
    "steps": 9,
    "seed": 0,
    "num_images_per_prompt": 1
  },
  "components": {},
  "quantization_config": null
}
```

Runnable scripts for these samples are under `examples/image_quantization_*.py`.

## Verification boundary

Common configuration profiles MUST pass a real model-file load and image
generation test on each target platform. Verification MUST record peak
VRAM, cold-load time, generation time, and repeated-result comparison at 512
by 512 and 1024 by 1024 where the pipeline supports those dimensions.

Repeated execution of one canonical task inside one validation pool MUST
satisfy the configured image perceptual-hash threshold.

The following observations establish baseline feasibility only:

- BitsAndBytes NF4 4-bit and INT8 produced byte-identical repeated images on
  Windows CUDA;
- floating-point LoRA was injected into a BitsAndBytes `Linear4bit` base and
  generated an image without fusing; and
- Nunchaku Lite executed on Linux CUDA with the memory and perceptual-hash
  results specified above.

These observations MUST NOT be generalized to an untested configuration,
model-file format, component class, GPU architecture, operating system, or
dependency version. Test coverage MUST document compatibility and drive
adapter fixes; it MUST NOT become a runtime model or configuration allowlist.
