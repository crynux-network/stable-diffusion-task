import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from typing import List, cast

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import DatasetDict, load_dataset
from diffusers import (AutoencoderKL, DDPMScheduler, DiffusionPipeline,
                       StableDiffusionPipeline, UNet2DConditionModel)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import convert_state_dict_to_diffusers
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from sd_task import utils
from sd_task.cache import ModelCache
from sd_task.config import Config, get_config
from sd_task.task_args import FinetuneLoraTaskArgs
from sd_task.task_runner.inference_task.errors import wrap_model_load_error

from .download_url_dataset import download_dataset_from_url

_logger = get_logger(__name__, log_level="INFO")


def generate_model_key(args: FinetuneLoraTaskArgs):
    model_args = {
        "model_name": args.model.name,
        "model_revision": args.model.revision,
        "mixed_precision": args.mixed_precision,
    }
    if args.model.variant is not None:
        model_args["model_variant"] = args.model.variant

    model_args_str = json.dumps(
        model_args, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    key = hashlib.md5(model_args_str.encode("utf-8")).hexdigest()
    return key


def run_finetune_lora_task(
    args: FinetuneLoraTaskArgs,
    output_dir: str,
    config: Config | None = None,
    model_cache: ModelCache | None = None,
):
    if config is None:
        config = get_config()

    # enable deterministic algorithms
    if config.deterministic and utils.get_accelerator() == "cuda":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False

    cache_dir = config.data_dir.models.huggingface
    train_args = args.train_args

    accelerator = Accelerator(
        gradient_accumulation_steps=train_args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        project_dir=output_dir,
    )

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # If passed along, set the training seed now.
    if args.seed is not None:
        print(f"seed: {args.seed}")
        set_seed(args.seed, deterministic=True)

    # Handle the repository creation
    if accelerator.is_main_process:
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    def load_model():
        with wrap_model_load_error(config.local_files_only):
            noise_scheduler: DDPMScheduler = DDPMScheduler.from_pretrained(
                args.model.name,
                subfolder="scheduler",
                cache_dir=cache_dir,
                local_files_only=config.local_files_only,
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                args.model.name,
                subfolder="tokenizer",
                revision=args.model.revision,
                cache_dir=cache_dir,
                local_files_only=config.local_files_only,
            )
            tokenizer = cast(CLIPTokenizer, tokenizer)
            text_encoder = CLIPTextModel.from_pretrained(
                args.model.name,
                subfolder="text_encoder",
                revision=args.model.revision,
                cache_dir=cache_dir,
                local_files_only=config.local_files_only,
            )
            text_encoder = cast(CLIPTextModel, text_encoder)
            vae = AutoencoderKL.from_pretrained(
                args.model.name,
                subfolder="vae",
                revision=args.model.revision,
                variant=args.model.variant,
                cache_dir=cache_dir,
                local_files_only=config.local_files_only,
            )
            vae = cast(CLIPTextModel, vae)
            unet = UNet2DConditionModel.from_pretrained(
                args.model.name,
                subfolder="unet",
                revision=args.model.revision,
                variant=args.model.variant,
                cache_dir=cache_dir,
                local_files_only=config.local_files_only,
            )
            unet = cast(UNet2DConditionModel, unet)
        # freeze parameters of models to save more memory
        unet.requires_grad_(False)
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        # Freeze the unet parameters before adding adapters
        for param in unet.parameters():
            param.requires_grad_(False)

        unet_lora_config = LoraConfig(
            r=args.lora.rank,
            lora_alpha=args.lora.rank,
            init_lora_weights=args.lora.init_lora_weights,
            target_modules=args.lora.target_modules,
        )

        # Move unet, vae and text_encoder to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)
        text_encoder.to(accelerator.device, dtype=weight_dtype)

        # Add adapter and make sure the trainable params are in float32.
        unet.add_adapter(unet_lora_config)
        if args.mixed_precision == "fp16":
            # only upcast trainable parameters (LoRA) into fp32
            cast_training_params(unet, dtype=torch.float32)

        with wrap_model_load_error(config.local_files_only):
            pipeline = DiffusionPipeline.from_pretrained(
                args.model.name,
                safety_checker=None,
                revision=args.model.revision,
                variant=args.model.variant,
                torch_dtype=weight_dtype,
                cache_dir=cache_dir,
                local_files_only=config.local_files_only,
            )
        pipeline = pipeline.to(accelerator.device)
        return noise_scheduler, tokenizer, text_encoder, vae, unet, pipeline

    if model_cache is not None:
        key = generate_model_key(args)
        noise_scheduler, tokenizer, text_encoder, vae, unet, pipeline = (
            model_cache.load(key, load_model)
        )
    else:
        noise_scheduler, tokenizer, text_encoder, vae, unet, pipeline = load_model()

    lora_layers = list(filter(lambda p: p.requires_grad, unet.parameters()))
    print(f"lora layers length: {len(lora_layers)}")

    if train_args.scale_lr:
        train_args.learning_rate = (
            train_args.learning_rate
            * train_args.gradient_accumulation_steps
            * train_args.batch_size
            * accelerator.num_processes
        )

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=train_args.learning_rate,
        betas=(train_args.adam_args.beta1, train_args.adam_args.beta2),
        weight_decay=train_args.adam_args.weight_decay,
        eps=train_args.adam_args.epsilon,
    )

    if args.dataset.name is not None:
        dataset = load_dataset(
            args.dataset.name,
            args.dataset.config_name,
            cache_dir=cache_dir,
        )
        dataset = cast(DatasetDict, dataset)
    elif args.dataset.url is not None:
        cache_dir = config.data_dir.models.external
        url = str(args.dataset.url)
        dirname = hashlib.md5(url.encode("utf-8")).hexdigest()
        dataset_dir = os.path.join(cache_dir, dirname)
        os.makedirs(dataset_dir, exist_ok=True)
        dataset_path = download_dataset_from_url(url, dataset_dir)
        dataset = load_dataset(dataset_path, args.dataset.config_name)
        dataset = cast(DatasetDict, dataset)
    else:
        raise ValueError("Either dataset.name or dataset.url must be provided")

    column_names = dataset["train"].column_names
    image_column = args.dataset.image_column
    if image_column not in column_names:
        raise ValueError(
            f"args.image_column value '{image_column}' needs to be one of: {', '.join(column_names)}"
        )

    caption_column = args.dataset.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"args.caption_column value '{caption_column}' needs to be one of: {', '.join(column_names)}"
        )

    captions: List[str] = dataset["train"][caption_column]

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                train_args.resolution,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            (
                transforms.CenterCrop(train_args.resolution)
                if args.transforms.center_crop
                else transforms.RandomCrop(train_args.resolution)
            ),
            (
                transforms.RandomHorizontalFlip()
                if args.transforms.random_flip
                else transforms.Lambda(lambda x: x)
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples

    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()  # type: ignore
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}

    generator = torch.manual_seed(args.seed)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        set_seed(worker_seed)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=train_args.batch_size,
        num_workers=0,
        generator=generator,
        worker_init_fn=seed_worker,
    )

    # Scheduler and math around the number of training steps.
    override_num_train_steps = False
    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_args.gradient_accumulation_steps
    )
    if train_args.num_train_steps is None:
        train_args.num_train_steps = (
            train_args.num_train_epochs * num_update_steps_per_epoch
        )
        override_num_train_steps = True
    if train_args.max_train_steps is None:
        train_args.max_train_steps = (
            train_args.max_train_epochs * num_update_steps_per_epoch
        )
        override_max_train_steps = True

    lr_scheduler = get_scheduler(
        train_args.lr_scheduler.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=train_args.lr_scheduler.lr_warmup_steps
        * accelerator.num_processes,
        num_training_steps=train_args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / train_args.gradient_accumulation_steps
    )
    if override_num_train_steps:
        train_args.num_train_steps = (
            train_args.num_train_epochs * num_update_steps_per_epoch
        )
    if override_max_train_steps:
        train_args.max_train_steps = (
            train_args.max_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    train_args.num_train_epochs = math.ceil(
        train_args.num_train_steps / num_update_steps_per_epoch
    )
    train_args.max_train_epochs = math.ceil(
        train_args.max_train_steps / num_update_steps_per_epoch
    )

    # Train!
    total_batch_size = (
        train_args.batch_size
        * accelerator.num_processes
        * train_args.gradient_accumulation_steps
    )

    _logger.info("***** Running training *****")
    _logger.info(f"  Num examples = {len(train_dataset)}")
    _logger.info(f"  Num Epochs = {train_args.num_train_epochs}")
    _logger.info(f"  Instantaneous batch size per device = {train_args.batch_size}")
    _logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    _logger.info(
        f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}"
    )
    _logger.info(f"  Optimization steps of this task = {train_args.num_train_steps}")
    _logger.info(f"  Total optimization steps = {train_args.max_train_steps}")
    step = 0
    global_step = 0
    first_epoch = 0

    initial_global_step = 0
    if args.checkpoint is not None:
        accelerator.print(f"Resuming from checkpoint {args.checkpoint}")
        # don't load the models' state
        _models = accelerator._models
        accelerator._models = []
        accelerator.load_state(args.checkpoint)
        accelerator._models = _models
        with open(os.path.join(args.checkpoint, "global_step.txt"), mode="r") as f:
            global_step = int(f.read())

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, initial_global_step + train_args.num_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, train_args.max_train_epochs):
        unet.train()
        train_loss = 0.0
        for batch in train_dataloader:
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(
                    batch["pixel_values"].to(dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if train_args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += train_args.noise_offset * torch.randn(
                        (latents.shape[0], latents.shape[1], 1, 1),
                        device=latents.device,
                    )

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config["num_train_timesteps"],
                    (bsz,),
                    device=latents.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(
                    batch["input_ids"], return_dict=False
                )[0]

                # Get the target for loss depending on the prediction type
                if train_args.prediction_type is not None:
                    # set prediction_type of scheduler if defined
                    noise_scheduler.register_to_config(
                        prediction_type=train_args.prediction_type
                    )

                if noise_scheduler.config["prediction_type"] == "epsilon":
                    target = noise
                elif noise_scheduler.config["prediction_type"] == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)  # type: ignore
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config['prediction_type']}"
                    )

                # Predict the noise residual and compute loss
                model_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states, return_dict=False
                )[0]

                if train_args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    mse_loss_weights = torch.stack(
                        [snr, train_args.snr_gamma * torch.ones_like(timesteps)], dim=1
                    ).min(dim=1)[0]
                    if noise_scheduler.config["prediction_type"] == "epsilon":
                        mse_loss_weights = mse_loss_weights / snr
                    elif noise_scheduler.config["prediction_type"] == "v_prediction":
                        mse_loss_weights = mse_loss_weights / (snr + 1)

                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_args.batch_size)).mean()  # type: ignore
                train_loss += avg_loss.item() / train_args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(
                        params_to_clip, train_args.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                step += 1
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if step >= train_args.num_train_steps:
                break

            if global_step >= train_args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(output_dir, "checkpoint")
        # don't save the model state to save disk space
        _models = accelerator._models
        accelerator._models = []
        accelerator.save_state(save_path)
        accelerator._models = _models

        unwrapped_unet = unwrap_model(unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(
            get_peft_model_state_dict(unwrapped_unet)
        )

        StableDiffusionPipeline.save_lora_weights(
            save_directory=save_path,
            unet_lora_layers=unet_lora_state_dict,
            safe_serialization=True,
        )

        with open(os.path.join(save_path, "global_step.txt"), mode="w") as f:
            f.write(str(global_step))

        if global_step >= train_args.max_train_steps:
            with open(os.path.join(save_path, "FINISH"), mode="w") as f:
                f.write("")

        _logger.info(f"Saved state to {save_path}")

        # Final inference
        # Load previous pipeline
        if args.validation.prompt is not None:
            prompts = [args.validation.prompt] * args.validation.num_images
        else:
            k = min(args.validation.num_images, len(captions))
            random.seed(args.seed)
            prompts = random.sample(captions, k)

        pipeline.load_lora_weights(save_path)

        # run inference
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator = generator.manual_seed(args.seed)
        images = []
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            for prompt in prompts:
                images.append(
                    pipeline(
                        prompt, num_inference_steps=30, generator=generator
                    ).images[0]
                )

        image_dir = os.path.join(output_dir, "validation")
        if not os.path.exists(image_dir):
            os.makedirs(image_dir, exist_ok=True)

        for i, image in enumerate(images):
            image_name = os.path.join(image_dir, f"{i}.png")
            image.save(image_name)

        pipeline.unload_lora_weights()

    accelerator.end_training()

    if utils.get_accelerator() == "cuda":
        torch.cuda.empty_cache()
