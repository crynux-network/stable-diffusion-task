from sd_task.config import Config, get_config
from sd_task.download_model import check_and_download_hf_pipeline
from sd_task.log import log


def prefetch_models(config: Config | None = None):
    if config is None:
        config = get_config()

    # base models
    if config.preloaded_models.base is not None:
        for model_config in config.preloaded_models.base:
            log("Preloading base model: " + model_config.id)

            call_args = {
                "hf_model_cache_dir": config.data_dir.models.huggingface,
                "proxy": config.proxy,
            }

            check_and_download_hf_pipeline(model_config.id, **call_args)
            log("Successfully preloaded base model: " + model_config.id)


if __name__ == "__main__":
    prefetch_models()
