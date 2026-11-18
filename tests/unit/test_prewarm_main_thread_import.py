"""The native ``torch`` import must happen on the MAIN thread.

``torch._C`` is a single-phase-init C extension. Importing such a module from a
non-main thread is a documented CPython fragility and is what made a
``patch.dict("sys.modules")`` eviction lethal (libtorch re-running
``initModule()`` -> SIGSEGV / exit 139, see PR #92). ``src.asgi._prewarm_kokoro``
therefore imports ``torch``/``kokoro`` in the coroutine body (main thread) and
only hands the heavy *pipeline construction* to ``asyncio.to_thread``.

This module guards that ordering: if the import is ever "tidied" back into the
worker thread, the first recorded import thread stops being ``MainThread`` and
these tests go red.

The observation hook wraps ``builtins.__import__`` rather than mutating
``sys.modules``. That is deliberate: the IMPORT_NAME opcode calls ``__import__``
on every ``import torch`` even when ``torch`` is already cached, so the hook sees
the statement without ever evicting the real extension module -- i.e. without
re-creating the exact eviction that caused the segfault this test exists for.
"""

from __future__ import annotations

import builtins
import threading
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from src.asgi import _prewarm_kokoro

# Module names whose import thread we care about.
_WATCHED = ("torch", "kokoro")


class _ImportThreadRecorder:
    """Records the thread on which each watched module is imported.

    Watched modules are satisfied with a stand-in module object so the test is
    hermetic and fast whether or not the optional ``tts-service[kokoro]`` extra
    is installed on this machine. Every other import is delegated untouched.
    """

    def __init__(self) -> None:
        self.records: list[tuple[str, str]] = []
        self._real = builtins.__import__

    def __call__(self, name: str, *args: Any, **kwargs: Any) -> Any:
        root = name.split(".")[0]
        if root in _WATCHED:
            self.records.append((root, threading.current_thread().name))
            stub = ModuleType(root)
            # `from kokoro import KPipeline` needs the attribute to exist.
            setattr(stub, "KPipeline", object)  # noqa: B010
            return stub
        return self._real(name, *args, **kwargs)

    def first_thread_for(self, module: str) -> str | None:
        for recorded, thread_name in self.records:
            if recorded == module:
                return thread_name
        return None


@pytest.fixture
def import_recorder(monkeypatch: pytest.MonkeyPatch) -> _ImportThreadRecorder:
    recorder = _ImportThreadRecorder()
    monkeypatch.setattr(builtins, "__import__", recorder)
    return recorder


@pytest.fixture
def kokoro_registry(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    """A registry whose kokoro provider records the thread it is built on.

    The stub ``_get_pipeline`` re-imports ``torch``/``kokoro`` exactly as the
    real one does, so that if the production import is moved back into the
    worker thread the recorder observes a worker thread name rather than
    observing nothing at all.
    """
    import src.asgi as asgi_mod
    from src.providers.kokoro_provider import KokoroProvider

    monkeypatch.setattr(asgi_mod.settings.kokoro, "enabled", True)

    build_threads: list[str] = []

    def _fake_get_pipeline(lang_code: str) -> object:
        build_threads.append(threading.current_thread().name)
        import torch  # noqa: F401
        from kokoro import KPipeline  # noqa: F401

        return object()

    monkeypatch.setattr(KokoroProvider, "_get_pipeline", staticmethod(_fake_get_pipeline))

    provider = KokoroProvider()
    registry = SimpleNamespace(
        provider_names={"kokoro"},
        get_provider=lambda name: provider,
    )
    return registry, build_threads


@pytest.mark.asyncio
async def test_prewarm_imports_torch_on_the_main_thread(
    import_recorder: _ImportThreadRecorder,
    kokoro_registry: tuple[Any, list[str]],
) -> None:
    registry, _ = kokoro_registry

    await _prewarm_kokoro(registry)

    first = import_recorder.first_thread_for("torch")
    assert first is not None, "torch was never imported during prewarm"
    assert first == "MainThread", (
        f"torch was first imported on {first!r}, not MainThread. "
        "The native import must stay in the _prewarm_kokoro coroutine body, "
        "outside asyncio.to_thread."
    )


@pytest.mark.asyncio
async def test_prewarm_imports_kokoro_on_the_main_thread(
    import_recorder: _ImportThreadRecorder,
    kokoro_registry: tuple[Any, list[str]],
) -> None:
    registry, _ = kokoro_registry

    await _prewarm_kokoro(registry)

    first = import_recorder.first_thread_for("kokoro")
    assert first is not None, "kokoro was never imported during prewarm"
    assert first == "MainThread", f"kokoro was first imported on {first!r}, not MainThread"


@pytest.mark.asyncio
async def test_prewarm_still_builds_the_pipeline_off_the_main_thread(
    import_recorder: _ImportThreadRecorder,
    kokoro_registry: tuple[Any, list[str]],
) -> None:
    """Only the import moved; the heavy construction stays in a worker thread."""
    registry, build_threads = kokoro_registry

    await _prewarm_kokoro(registry)

    assert build_threads, "_get_pipeline was never called"
    assert build_threads[0] != "MainThread", (
        "pipeline construction ran on the main thread — it must stay in "
        "asyncio.to_thread so startup is not blocked synchronously"
    )


@pytest.mark.asyncio
async def test_prewarm_tolerates_missing_torch_extra(
    monkeypatch: pytest.MonkeyPatch,
    kokoro_registry: tuple[Any, list[str]],
) -> None:
    """A deployment without ``tts-service[kokoro]`` must still start."""
    registry, _ = kokoro_registry
    real_import = builtins.__import__

    def _no_torch(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.split(".")[0] in _WATCHED:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_torch)

    # Must not raise: the existing (ImportError, OSError, RuntimeError, ValueError)
    # handler swallows it, exactly as before the import was hoisted.
    await _prewarm_kokoro(registry)
