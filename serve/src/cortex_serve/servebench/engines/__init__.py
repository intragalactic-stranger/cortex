"""Engine launchers. Each builds a ``docker run`` command, starts it, waits for health, and stops it."""

from cortex_serve.servebench.engines.base import Engine
from cortex_serve.servebench.engines.llamacpp import LlamaCppEngine
from cortex_serve.servebench.engines.ollama import OllamaEngine
from cortex_serve.servebench.engines.process import ProcessEngine

__all__ = ["Engine", "LlamaCppEngine", "OllamaEngine", "ProcessEngine", "make_engine"]


def make_engine(kind: str, image: str, args: dict[str, object], models_dir: str, launch: str = "docker") -> Engine:
    if launch == "process":
        return ProcessEngine(kind, image, dict(args), models_dir)
    if kind == "llamacpp":
        return LlamaCppEngine(image, args, models_dir)
    if kind == "ollama":
        return OllamaEngine(image, args, models_dir)
    raise ValueError(f"unsupported engine kind {kind!r}")
