"""Named collection of torch modules with save/load and partial weight exchange."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

import torch
from torch import nn


class ModelBundle:
    def __init__(self, modules: dict[str, nn.Module]):
        self._modules: dict[str, nn.Module] = dict(modules)

    # ------------------------------------------------------------------
    # Container behaviour
    # ------------------------------------------------------------------
    def __getitem__(self, name: str) -> nn.Module:
        return self._modules[name]

    def __contains__(self, name: str) -> bool:
        return name in self._modules

    def __iter__(self) -> Iterator[str]:
        return iter(self._modules)

    @property
    def names(self) -> list[str]:
        return list(self._modules.keys())

    def items(self):
        return self._modules.items()

    def parameters(self, names: Optional[Iterable[str]] = None):
        for name in (names if names is not None else self._modules):
            yield from self._modules[name].parameters()

    # ------------------------------------------------------------------
    # Device / mode
    # ------------------------------------------------------------------
    def to(self, device: str | torch.device) -> "ModelBundle":
        for module in self._modules.values():
            module.to(device)
        return self

    def eval(self) -> "ModelBundle":
        for module in self._modules.values():
            module.eval()
        return self

    def train(self) -> "ModelBundle":
        for module in self._modules.values():
            module.train()
        return self

    # ------------------------------------------------------------------
    # Weight exchange
    # ------------------------------------------------------------------
    def state_dicts(self, names: Optional[Iterable[str]] = None) -> dict[str, dict[str, Any]]:
        """CPU state dicts keyed by module name (all modules if ``names`` is None)."""
        selected = list(names) if names is not None else self.names
        out: dict[str, dict[str, Any]] = {}
        for name in selected:
            sd = self._modules[name].state_dict()
            out[name] = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in sd.items()}
        return out

    def load_state_dicts(self, sds: dict[str, dict[str, Any]], strict: bool = True) -> list[str]:
        """Load the given module state dicts; unknown module names are ignored. Returns loaded names."""
        loaded: list[str] = []
        for name, sd in sds.items():
            module = self._modules.get(name)
            if module is None:
                continue
            module.load_state_dict(sd, strict=strict)
            loaded.append(name)
        return loaded

    # ------------------------------------------------------------------
    # Disk
    # ------------------------------------------------------------------
    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        os.makedirs(model_dir, exist_ok=True)
        for name, sd in self.state_dicts().items():
            tmp = model_dir / f"{name}.pt.tmp"
            torch.save(sd, tmp)
            os.replace(tmp, model_dir / f"{name}.pt")

    def load(self, model_dir: str | Path, device: str | torch.device = "cpu", strict: bool = True) -> list[str]:
        """Load ``<name>.pt`` for every module that has a file. Returns loaded names."""
        model_dir = Path(model_dir)
        loaded: list[str] = []
        for name, module in self._modules.items():
            path = model_dir / f"{name}.pt"
            if not path.is_file():
                continue
            sd = torch.load(path, map_location=device, weights_only=True)
            module.load_state_dict(sd, strict=strict)
            loaded.append(name)
        return loaded
