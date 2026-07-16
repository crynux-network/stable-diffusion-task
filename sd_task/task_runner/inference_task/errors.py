from contextlib import contextmanager
from typing import Optional, Type, cast

from huggingface_hub.utils import (
    GatedRepoError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from requests import ConnectionError, HTTPError


from sd_task import utils


__all__ = [
    "wrap_download_error",
    "wrap_execution_error",
    "wrap_model_load_error",
    "ModelInvalid",
    "ModelDownloadError",
    "ModelNotDownloaded",
    "TaskExecutionError",
    "TaskVersionNotSupported",
]


class TaskVersionNotSupported(ValueError):
    def __str__(self) -> str:
        return "Task version not supported"


class ModelInvalid(ValueError):
    def __str__(self) -> str:
        return "Task model invalid"


class ModelDownloadError(ValueError):
    def __str__(self) -> str:
        return "Task model download error"


class ModelNotDownloaded(ValueError):
    def __str__(self) -> str:
        return "Task model not downloaded"


class TaskExecutionError(ValueError):
    def __str__(self) -> str:
        return "Task execution error"


def travel_exc(e: BaseException):
    queue = [e]
    exc_set = set(queue)

    while len(queue) > 0:
        exc = queue.pop(0)
        yield exc
        if exc.__cause__ is not None and exc.__cause__ not in exc_set:
            queue.append(exc.__cause__)
            exc_set.add(exc.__cause__)
        if exc.__context__ is not None and exc.__context__ not in exc_set:
            queue.append(exc.__context__)
            exc_set.add(exc.__context__)


def match_exception(
    e: Exception, target: Type[Exception], message: Optional[str] = None
) -> Optional[Exception]:
    for exc in travel_exc(e):
        if isinstance(exc, target) and (message is None or message in str(exc)):
            return exc
    return None


@contextmanager
def wrap_download_error():
    try:
        yield
    except Exception as e:
        if match_exception(e, LocalEntryNotFoundError):
            raise ModelDownloadError from e
        elif (
            match_exception(e, RepositoryNotFoundError)
            or match_exception(e, RevisionNotFoundError)
            or match_exception(e, GatedRepoError)
        ):
            raise ModelInvalid from e
        elif exc := match_exception(e, HTTPError):
            exc = cast(HTTPError, e)
            if exc.response is not None and 400 <= exc.response.status_code < 500:
                raise ModelInvalid from e

            raise ModelDownloadError from e
        elif match_exception(e, ConnectionError):
            raise ModelDownloadError from e
        raise ModelDownloadError from e


def _wrap_cuda_execution_error():
    from torch.cuda import OutOfMemoryError

    try:
        yield
    except OutOfMemoryError:
        raise
    except RuntimeError as e:
        if "out of memory" in str(e):
            raise
        else:
            raise TaskExecutionError from e
    except Exception as e:
        raise TaskExecutionError from e


def _wrap_macos_execution_error():
    try:
        yield
    except Exception as e:
        raise TaskExecutionError from e


@contextmanager
def wrap_execution_error():
    if utils.get_accelerator() == "mps":
        yield from _wrap_macos_execution_error()
    else:
        yield from _wrap_cuda_execution_error()


# Wraps model loading in local_files_only mode: a load failure caused by a
# local cache miss is reported as ModelNotDownloaded, everything else keeps
# the wrap_execution_error classification.
@contextmanager
def wrap_model_load_error(local_files_only: bool = False):
    if not local_files_only:
        with wrap_execution_error():
            yield
        return

    try:
        yield
    except ModelNotDownloaded:
        raise
    except Exception as e:
        if match_exception(e, EnvironmentError) or match_exception(
            e, LocalEntryNotFoundError
        ):
            raise ModelNotDownloaded from e
        with wrap_execution_error():
            raise
