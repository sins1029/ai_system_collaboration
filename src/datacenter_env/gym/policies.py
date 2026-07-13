from __future__ import annotations

from typing import Any, Mapping

import numpy as np


class MaskedRandomPolicy:
    def __init__(self, seed: int | None = None):
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def action(
        self,
        observation: Mapping[str, np.ndarray],
        info: Mapping[str, Any] | None = None,
    ) -> int:
        del info
        mask = np.asarray(observation["action_mask"], dtype=np.int8)
        legal = np.flatnonzero(mask)
        if not len(legal):
            raise RuntimeError("masked random policy received no legal action")
        return int(self._rng.choice(legal))
