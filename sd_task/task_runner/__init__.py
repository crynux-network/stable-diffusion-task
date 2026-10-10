from .image_task import run_image_task

__all__ = [
    "run_image_task",
    "run_finetune_lora_task",
    "run_remove_background_task",
    "run_tagger_task",
    "run_controlnet_preprocess_task",
]


def __getattr__(name: str):
    if name == "run_finetune_lora_task":
        from .finetune_task import run_finetune_lora_task

        return run_finetune_lora_task
    if name == "run_remove_background_task":
        from .remove_background import run_remove_background_task

        return run_remove_background_task
    if name == "run_tagger_task":
        from .tagger import run_tagger_task

        return run_tagger_task
    if name == "run_controlnet_preprocess_task":
        from .controlnet_preprocess import run_controlnet_preprocess_task

        return run_controlnet_preprocess_task
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
