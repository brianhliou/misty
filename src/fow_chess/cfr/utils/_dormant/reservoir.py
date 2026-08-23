"""Reservoir-sampling buffer for memory-bounded streaming collection.

Parked here in Phase A0 of the Obscuro replication. Has no active
consumer — the Deep CFR code path that built it was deleted along
with the rest of the function-approximation substrate.

Likely future consumer: Phase C neural evaluator training, if/when
that work happens. Implements Vitter's reservoir sampling (Brown
et al. 2019 "Deep CFR" specifies this for memory-bounded sample
collection in CFR).
"""

from __future__ import annotations

import random


class ReservoirBuffer:
    """Reservoir-sampling buffer to bound memory of streaming sample
    collections.

    With ``max_size=None`` behaves as an unbounded list. With
    ``max_size=N``, keeps a uniform-random subset of the ``seen`` items
    via Vitter's reservoir sampling — once the buffer is full, each
    subsequent ``append`` replaces a random existing slot with
    probability ``max_size / seen``.
    """

    def __init__(
        self,
        max_size: int | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._buffer: list = []
        self._max = max_size
        self._rng = rng or random.Random()
        self._seen = 0

    def append(self, item) -> None:
        self._seen += 1
        if self._max is None or len(self._buffer) < self._max:
            self._buffer.append(item)
            return
        i = self._rng.randint(0, self._seen - 1)
        if i < self._max:
            self._buffer[i] = item

    def __iter__(self):
        return iter(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)
