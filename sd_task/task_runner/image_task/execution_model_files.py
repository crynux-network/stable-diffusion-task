from typing import Protocol, Sequence

from sd_task.task_args.image_task import ImageTaskArgs

from .errors import ExecutionModelFileError

BASE_MODEL_FORMAT = "diffusers"


class ExecutionModelFile(Protocol):
    repository: str
    revision: str
    format: str


def _is_commit_sha(revision: str) -> bool:
    if len(revision) != 40:
        return False
    try:
        int(revision, 16)
    except ValueError:
        return False
    return True


def _allocated_revision(
    model_files: Sequence[ExecutionModelFile],
    repository: str,
    format_name: str,
    configured_revision: str | None,
) -> str:
    entries = [
        entry
        for entry in model_files
        if entry.repository == repository and entry.format == format_name
    ]
    if not entries:
        raise ExecutionModelFileError(
            f"No execution model file is allocated for {format_name} model "
            f"{repository}"
        )

    revisions = {entry.revision for entry in entries}
    if len(revisions) != 1:
        raise ExecutionModelFileError(
            f"Conflicting execution model file revisions for model {repository}"
        )
    revision = revisions.pop()
    if revision == "":
        raise ExecutionModelFileError(
            f"Execution model file of model {repository} has no revision"
        )
    if (
        configured_revision is not None
        and _is_commit_sha(configured_revision)
        and configured_revision != revision
    ):
        raise ExecutionModelFileError(
            f"Task revision {configured_revision} conflicts with the allocated "
            f"revision {revision} of model {repository}"
        )
    return revision


def apply_execution_model_files(
    args: ImageTaskArgs, model_files: Sequence[ExecutionModelFile]
) -> ImageTaskArgs:
    """Return task args that load the exact revisions allocated for the models.

    The allocation is an independent load constraint matched by repository and
    format. Branch or tag values in TaskArgs do not override it. A TaskArgs
    commit SHA that differs from the allocation is a conflict.
    """
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
