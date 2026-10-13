from dataclasses import dataclass, field

import pytest

from sd_task.task_args import ImageTaskArgs
from sd_task.task_runner.image_task import (
    ExecutionModelFileError,
    apply_execution_model_files,
)
from sd_task.task_runner.image_task.key import generate_model_key

SHA = "a" * 40
OTHER_SHA = "b" * 40
COMPONENT_SHA = "c" * 40


@dataclass(frozen=True)
class Entry:
    repository: str
    revision: str
    format: str = "diffusers"
    format_kwargs: dict = field(default_factory=dict)


def make_args(**updates) -> ImageTaskArgs:
    data = {
        "version": "3.7.0",
        "base_model": {"name": "org/base"},
        "prompt": "test",
        "task_config": {},
    }
    data.update(updates)
    return ImageTaskArgs.model_validate(data)


def gguf_component(**updates) -> dict:
    component = {
        "format": "gguf",
        "repo": "org/gguf",
        "format_kwargs": {"filename": "model-Q4_0.gguf"},
    }
    component.update(updates)
    return {"transformer": component}


def gguf_entry(revision: str = COMPONENT_SHA) -> Entry:
    return Entry("org/gguf", revision, "gguf", {"filename": "model-Q4_0.gguf"})


def test_applies_base_model_revision():
    args = make_args()

    applied = apply_execution_model_files(args, [Entry("org/base", SHA)])

    assert applied.base_model.revision == SHA
    assert args.base_model.revision is None


def test_applies_component_revision():
    args = make_args(components=gguf_component())

    applied = apply_execution_model_files(
        args, [Entry("org/base", SHA), gguf_entry()]
    )

    assert applied.base_model.revision == SHA
    assert applied.components["transformer"].revision == COMPONENT_SHA
    assert args.components["transformer"].revision is None


def test_accepts_matching_task_revisions():
    args = make_args(
        base_model={"name": "org/base", "revision": SHA},
        components=gguf_component(revision=COMPONENT_SHA),
    )

    applied = apply_execution_model_files(
        args, [Entry("org/base", SHA), gguf_entry()]
    )

    assert applied.base_model.revision == SHA


def test_rejects_conflicting_base_revision():
    args = make_args(base_model={"name": "org/base", "revision": OTHER_SHA})

    with pytest.raises(ExecutionModelFileError):
        apply_execution_model_files(args, [Entry("org/base", SHA)])


def test_branch_task_revision_uses_allocated_sha():
    args = make_args(base_model={"name": "org/base", "revision": "main"})

    applied = apply_execution_model_files(args, [Entry("org/base", SHA)])

    assert applied.base_model.revision == SHA
    assert args.base_model.revision == "main"


def test_rejects_conflicting_component_revision():
    args = make_args(components=gguf_component(revision=OTHER_SHA))

    with pytest.raises(ExecutionModelFileError):
        apply_execution_model_files(args, [Entry("org/base", SHA), gguf_entry()])


def test_rejects_missing_allocation():
    with pytest.raises(ExecutionModelFileError):
        apply_execution_model_files(make_args(), [])
    with pytest.raises(ExecutionModelFileError):
        apply_execution_model_files(
            make_args(components=gguf_component()), [Entry("org/base", SHA)]
        )


def test_rejects_empty_revision():
    with pytest.raises(ExecutionModelFileError):
        apply_execution_model_files(make_args(), [Entry("org/base", "")])


def test_rejects_conflicting_allocations():
    with pytest.raises(ExecutionModelFileError):
        apply_execution_model_files(
            make_args(), [Entry("org/base", SHA), Entry("org/base", OTHER_SHA)]
        )


def test_matches_entries_by_format():
    with pytest.raises(ExecutionModelFileError):
        apply_execution_model_files(make_args(), [Entry("org/base", SHA, "gguf")])


def test_model_key_depends_on_allocated_revision():
    class Pipeline:
        pass

    first = apply_execution_model_files(make_args(), [Entry("org/base", SHA)])
    second = apply_execution_model_files(make_args(), [Entry("org/base", OTHER_SHA)])

    assert generate_model_key(first, Pipeline) != generate_model_key(second, Pipeline)
