import hashlib
import os
from contextlib import contextmanager
from typing import Callable, Union

import requests
import validators
from diffusers import DiffusionPipeline
from huggingface_hub import hf_hub_download, model_info
from huggingface_hub.utils import EntryNotFoundError
from tqdm import tqdm

from sd_task.config import ProxyConfig
from sd_task.log import log


def check_and_download_model_by_name(
    model_name: str,
    loader: Union[Callable, None],
    weights_names: list[str],
    guess_weights_name: bool,
    **kwargs,
) -> tuple[str, str]:
    hf_model_cache_dir = kwargs.pop("hf_model_cache_dir")
    external_model_cache_dir = kwargs.pop("external_model_cache_dir")
    proxy = kwargs.pop("proxy")
    variant = kwargs.pop("variant", None)

    if validators.url(model_name):
        return check_and_download_external_model(
            model_name, external_model_cache_dir, proxy
        )
    else:
        return check_and_download_hf_model(
            model_name,
            loader,
            weights_names,
            guess_weights_name,
            hf_model_cache_dir,
            proxy,
            variant,
        )


def check_and_download_external_model(
    model_name: str, external_cache_dir: str, proxy: ProxyConfig | None
) -> tuple[str, str]:

    log("Check and download the external model file: " + model_name)

    weight_file_name = "model.safetensors"

    m = hashlib.sha256()
    m.update(model_name.encode("utf-8"))
    url_hash = m.hexdigest()

    model_folder = os.path.join(external_cache_dir, url_hash)
    model_file = os.path.join(model_folder, weight_file_name)

    log("The model file will be saved as: " + model_file)

    # Check if we have already cached the model file
    if os.path.isdir(model_folder):
        if os.path.isfile(model_file):
            log("Found a local cache of the model file. Skip the download")
            return model_file, weight_file_name
    else:
        os.makedirs(model_folder, mode=0o755, exist_ok=True)

    # Download the model file
    model_file = os.path.join(model_folder, "model.safetensors")

    log("Model file not cached locally. Start the download...")

    try:
        with requests_proxy_session(proxy) as proxies:
            resp = requests.get(model_name, stream=True, timeout=5, proxies=proxies)

            resp.raise_for_status()

            total_bytes = int(resp.headers.get("content-length", 0))

            with tqdm.wrapattr(
                open(model_file, "wb"),
                "write",
                miniters=1,
                desc=model_file,
                total=total_bytes,
            ) as f_out:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        f_out.write(chunk)
                f_out.flush()

        return model_folder, weight_file_name
    except Exception as e:
        # delete the broken file if download failed
        if os.path.isfile(model_file):
            os.remove(model_file)

        raise e


def check_and_download_hf_pipeline(
    model_name: str, variant: str | None, **kwargs
) -> str:
    log("Check and download the Huggingface pipeline: " + model_name)

    hf_model_cache_dir = kwargs.pop("hf_model_cache_dir")
    proxy = kwargs.pop("proxy")

    def _download(call_args):
        with requests_proxy_session(proxy):
            DiffusionPipeline.download(model_name, **call_args)

    try:
        call_args = {
            "cache_dir": hf_model_cache_dir,
            "variant": variant,
        }
        _download(call_args)
    except ValueError as e:
        if "no variant default" in str(e):
            call_args = {
                "cache_dir": hf_model_cache_dir,
            }
            _download(call_args)
        else:
            raise e

    return model_name


def check_and_download_hf_model(
    model_name: str,
    config_loader: Union[Callable, None],
    weights_names: list[str],
    guess_weight_name: bool,
    hf_model_cache_dir: str,
    proxy: ProxyConfig | None,
    variant: str | None = None,
) -> tuple[str, str]:
    log("Check and download the Huggingface model file: " + model_name)
    weight_file_name = ""

    with requests_proxy_session(proxy):
        # download the config file
        call_args = {
            "cache_dir": hf_model_cache_dir,
        }
        if config_loader is not None:
            config_loader(model_name, **call_args)

        model_file = None

        for weights_name in weights_names:
            if model_file is None:
                try:
                    if variant is not None:
                        call_args["filename"] = add_variant(weights_name, variant)
                    else:
                        call_args["filename"] = weights_name
                    model_file = hf_hub_download(model_name, **call_args)
                    weight_file_name = call_args["filename"]
                except EntryNotFoundError:
                    pass

        if model_file is None:
            for idx, weights_name in enumerate(weights_names):

                if model_file is None:

                    call_args["filename"] = weights_name

                    if (not guess_weight_name) and idx == len(weights_names) - 1:
                        model_file = hf_hub_download(model_name, **call_args)
                        weight_file_name = call_args["filename"]
                    else:
                        try:
                            model_file = hf_hub_download(model_name, **call_args)
                            weight_file_name = call_args["filename"]
                        except EntryNotFoundError:
                            pass

        if model_file is None and guess_weight_name:
            weight_name = best_guess_weight_name(
                model_name, file_extension=".safetensors"
            )

            if weight_name is None:
                weight_name = best_guess_weight_name(model_name, file_extension=".bin")

                if weight_name is None:
                    # To raise the same EntryNotFound Error
                    hf_hub_download(model_name, **call_args)
                    return model_name, ""

            call_args["filename"] = weight_name
            hf_hub_download(model_name, **call_args)
            weight_file_name = weight_name

    return model_name, weight_file_name


def get_requests_proxy_url(proxy: ProxyConfig | None) -> str | None:
    if proxy is not None and proxy.host != "":

        if "://" in proxy.host:
            scheme, host = proxy.host.split("://", 2)
        else:
            scheme, host = "", proxy.host

        proxy_str = ""
        if scheme != "":
            proxy_str += f"{scheme}://"

        if proxy.username != "":
            proxy_str += f"{proxy.username}"

            if proxy.password != "":
                proxy_str += f":{proxy.password}"

            proxy_str += f"@"

        proxy_str += f"{host}:{proxy.port}"

        return proxy_str
    else:
        return None


@contextmanager
def requests_proxy_session(proxy: ProxyConfig | None):
    proxy_url = get_requests_proxy_url(proxy)
    if proxy_url is not None:
        origin_http_proxy = os.environ.get("HTTP_PROXY", None)
        origin_https_proxy = os.environ.get("HTTPS_PROXY", None)
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        try:
            yield {
                "http": proxy_url,
                "https": proxy_url,
            }
        finally:
            if origin_http_proxy is not None:
                os.environ["HTTP_PROXY"] = origin_http_proxy
            else:
                os.environ.pop("HTTP_PROXY")
            if origin_https_proxy is not None:
                os.environ["HTTPS_PROXY"] = origin_https_proxy
            else:
                os.environ.pop("HTTPS_PROXY")
    else:
        yield None


def add_variant(weights_name: str, variant: str | None = None) -> str:
    if variant is not None:
        splits = weights_name.split(".")
        splits = splits[:-1] + [variant] + splits[-1:]
        weights_name = ".".join(splits)

    return weights_name


def best_guess_weight_name(
    pretrained_model_name_or_path_or_dict, file_extension=".safetensors"
) -> str | None:

    files_in_repo = model_info(pretrained_model_name_or_path_or_dict).siblings
    targeted_files = [
        f.rfilename for f in files_in_repo if f.rfilename.endswith(file_extension)
    ]

    if len(targeted_files) == 0:
        return

    disallowed_substrings = {"scheduler", "optimizer", "checkpoint"}
    targeted_files = list(
        filter(
            lambda x: all(substring not in x for substring in disallowed_substrings),
            targeted_files,
        )
    )

    if len(targeted_files) > 1:
        raise ValueError(
            f"Provided path contains more than one weights file in the {file_extension} format. Either specify "
            f"`weight_name` in `load_lora_weights` or make sure there's only one  `.safetensors` or `.bin` file in  "
            f"{pretrained_model_name_or_path_or_dict}."
        )
    weight_name = targeted_files[0]
    return weight_name
