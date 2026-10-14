from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sd_task.task_args import ImageTaskArgs
from sd_task.task_args.image_task import ComponentSource
from sd_task.task_runner.image_task import (
    apply_execution_model_files,
)
from sd_task.task_runner.image_task import pipeline as pipeline_module
from sd_task.task_runner.image_task.errors import TaskInvalid
from sd_task.task_runner.image_task.quantization.diffusers_component import (
    load_diffusers_component,
    resolve_diffusers_subfolder,
)
from sd_task.task_runner.image_task.quantization.errors import (
    UnsupportedComponentFormat,
)
from sd_task.task_runner.image_task.quantization.registry import (
    resolve_component_loader,
)

SHA = "a" * 40
COMPONENT_SHA = "c" * 40


def make_args(**updates) -> ImageTaskArgs:
    data = {
        "version": "3.7.0",
        "base_model": {"name": "org/base"},
        "prompt": "test",
        "task_config": {},
    }
    data.update(updates)
    return ImageTaskArgs.model_validate(data)


def diffusers_vae(**updates) -> dict:
    component = {
        "format": "diffusers",
        "repo": "org/vae",
        "format_kwargs": {},
    }
    component.update(updates)
    return {"vae": component}


def test_registry_accepts_diffusers_component():
    source = ComponentSource(format="diffusers", repo="org/vae")
    assert resolve_component_loader(source) == "diffusers"


def test_registry_accepts_diffusers_subfolder():
    source = ComponentSource(
        format="diffusers",
        repo="org/repo",
        format_kwargs={"subfolder": "vae"},
    )
    assert resolve_component_loader(source) == "diffusers"
    assert resolve_diffusers_subfolder(source) == "vae"


def test_registry_rejects_empty_subfolder():
    source = ComponentSource(
        format="diffusers",
        repo="org/repo",
        format_kwargs={"subfolder": ""},
    )
    with pytest.raises(TaskInvalid):
        resolve_component_loader(source)


def test_registry_rejects_unknown_format():
    source = ComponentSource(format="other", repo="org/vae")
    with pytest.raises(UnsupportedComponentFormat):
        resolve_component_loader(source)


def test_load_diffusers_component_uses_model_index_class(monkeypatch):
    loaded = object()

    class FakeVAE:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            assert path == "/cache/org/vae"
            assert kwargs["local_files_only"] is True
            assert kwargs["torch_dtype"] is torch.float16
            assert "subfolder" not in kwargs
            return loaded

    monkeypatch.setattr(
        "sd_task.task_runner.image_task.quantization.diffusers_component."
        "component_class_from_pipeline_config",
        lambda config, name: FakeVAE,
    )

    result = load_diffusers_component(
        pipeline_config={"vae": ["diffusers", "AutoencoderKL"]},
        component_name="vae",
        source=ComponentSource(format="diffusers", repo="org/vae"),
        cache_dir="/cache",
        torch_dtype=torch.float16,
        local_files_only=True,
        component_path="/cache/org/vae",
    )
    assert result is loaded


def test_load_diffusers_component_rejects_missing_model_index_entry():
    with pytest.raises(TaskInvalid, match="does not declare vae"):
        load_diffusers_component(
            pipeline_config={},
            component_name="vae",
            source=ComponentSource(format="diffusers", repo="org/vae"),
            cache_dir="/cache",
            torch_dtype=None,
            local_files_only=True,
            component_path="/cache/org/vae",
        )


def test_load_pipeline_injects_diffusers_component(monkeypatch):
    injected = object()
    captured = {}

    class FakePipeline:
        @classmethod
        def from_pretrained(cls, name, **kwargs):
            captured["name"] = name
            captured["kwargs"] = kwargs
            return SimpleNamespace(name=name)

    monkeypatch.setattr(
        "sd_task.task_runner.image_task.quantization.gguf.pipeline_config_from_base",
        lambda *args, **kwargs: {"vae": ["diffusers", "AutoencoderKL"]},
    )
    monkeypatch.setattr(
        "sd_task.task_runner.image_task.quantization.diffusers_component."
        "load_diffusers_component",
        lambda **kwargs: injected,
    )

    args = make_args(components=diffusers_vae())
    result = pipeline_module.load_pipeline(
        FakePipeline,
        args,
        cache_dir="/cache",
        torch_dtype=torch.float16,
        local_files_only=True,
        component_paths={"vae": "/cache/org/vae"},
    )

    assert result.name == "org/base"
    assert captured["kwargs"]["vae"] is injected
    assert captured["kwargs"]["dtype"] is torch.float16


def test_load_pipeline_rejects_gguf_non_transformer(monkeypatch):
    monkeypatch.setattr(
        "sd_task.task_runner.image_task.quantization.gguf.pipeline_config_from_base",
        lambda *args, **kwargs: {"vae": ["diffusers", "AutoencoderKL"]},
    )
    args = make_args(
        components={
            "vae": {
                "format": "gguf",
                "repo": "org/gguf",
                "format_kwargs": {"filename": "vae.gguf"},
            }
        }
    )
    with pytest.raises(TaskInvalid, match="GGUF adapter does not support"):
        pipeline_module.load_pipeline(
            MagicMock(),
            args,
            cache_dir="/cache",
            torch_dtype=None,
            local_files_only=True,
            component_paths={"vae": "/cache/vae.gguf"},
        )


def test_applies_diffusers_component_revision():
    from dataclasses import dataclass, field

    @dataclass(frozen=True)
    class Entry:
        repository: str
        revision: str
        format: str = "diffusers"
        format_kwargs: dict = field(default_factory=dict)

    args = make_args(components=diffusers_vae())
    applied = apply_execution_model_files(
        args,
        [
            Entry("org/base", SHA),
            Entry("org/vae", COMPONENT_SHA, "diffusers"),
        ],
    )
    assert applied.components["vae"].revision == COMPONENT_SHA
    assert args.components["vae"].revision is None


def test_download_diffusers_component_uses_snapshot(monkeypatch):
    from sd_task.task_runner.image_task import download_model as download_module

    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        return "/cache/snapshots/org/vae"

    monkeypatch.setattr(download_module, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(
        download_module,
        "requests_proxy_session",
        lambda proxy: MagicMock(
            __enter__=lambda self: None, __exit__=lambda *args: None
        ),
    )

    path = download_module.download_component_file(
        ComponentSource(
            format="diffusers",
            repo="org/vae",
            revision=COMPONENT_SHA,
            format_kwargs={"subfolder": "vae"},
        ),
        "/cache",
        None,
    )
    assert path == "/cache/snapshots/org/vae"
    assert calls["repo_id"] == "org/vae"
    assert calls["revision"] == COMPONENT_SHA
    assert calls["allow_patterns"] == ["vae/*", "vae/**"]
