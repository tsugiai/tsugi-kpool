"""tsugi-kpool: K-Pool LoRA SDK.

Public API surface kept intentionally small. See README.md for usage.

Note: KPoolLoraConfig is pure-python and imports without torch installed
(so tests / configuration / docs tooling work in lighter environments).
plesio_init and plesio_shutdown require torch at import time.
"""
from tsugi_kpool.config import KPoolLoraConfig

__version__ = "0.1.2"


def plesio_init(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tsugi_kpool.runtime import plesio_init as _impl
    return _impl(*args, **kwargs)


def plesio_shutdown(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tsugi_kpool.runtime import plesio_shutdown as _impl
    return _impl(*args, **kwargs)


def apply_kpool_step(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tsugi_kpool.runtime import apply_kpool_step as _impl
    return _impl(*args, **kwargs)


def pre_forward_step(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tsugi_kpool.runtime import pre_forward_step as _impl
    return _impl(*args, **kwargs)


def post_backward_step(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tsugi_kpool.runtime import post_backward_step as _impl
    return _impl(*args, **kwargs)


def get_runtime(*args, **kwargs):  # type: ignore[no-untyped-def]
    from tsugi_kpool.runtime import get_runtime as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "KPoolLoraConfig",
    "plesio_init",
    "plesio_shutdown",
    "apply_kpool_step",
    "pre_forward_step",
    "post_backward_step",
    "get_runtime",
    "__version__",
]
