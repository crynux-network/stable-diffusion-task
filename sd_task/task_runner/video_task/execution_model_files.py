from typing import Sequence

from sd_task.task_args.video_task import VideoTaskArgs
from sd_task.task_runner.image_task.execution_model_files import (
    BASE_MODEL_FORMAT,
    ExecutionModelFile,
    _allocated_revision,
)


def apply_execution_model_files(
    args: VideoTaskArgs, model_files: Sequence[ExecutionModelFile]
) -> VideoTaskArgs:
    """Return video task args pinned to the exact revisions allocated by Relay."""
    applied = args.model_copy(deep=True)
    applied.base_model.revision = _allocated_revision(
        model_files,
        args.base_model.name,
        BASE_MODEL_FORMAT,
        args.base_model.revision,
    )
    for source in applied.components.values():
        source.revision = _allocated_revision(
            model_files, source.repo, source.format, source.revision
        )
    return applied
