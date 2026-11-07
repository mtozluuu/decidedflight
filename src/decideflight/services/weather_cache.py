"""Bounded request sharing; failed loads never reuse expired data."""

import asyncio
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Hashable
from copy import deepcopy
import time
from typing import Generic, TypeVar

T = TypeVar("T")


class WeatherCache(Generic[T]):
    def __init__(self, ttl: float, max_entries: int = 128):
        self.ttl = ttl
        self.max_entries = max_entries
        self.entries: OrderedDict[Hashable, tuple[float, T]] = OrderedDict()
        self.pending: dict[
            tuple[asyncio.AbstractEventLoop, Hashable], asyncio.Task[T]
        ] = {}

    async def get(self, key: Hashable, load: Callable[[], Awaitable[T]]) -> T:
        cached = self.entries.get(key)
        if cached and time.monotonic() - cached[0] < self.ttl:
            self.entries.move_to_end(key)
            return deepcopy(cached[1])
        self.entries.pop(key, None)
        task_key = (asyncio.get_running_loop(), key)
        task = self.pending.get(task_key)
        if task is None:

            async def fetch() -> T:
                try:
                    result = await load()
                    self.entries[key] = (time.monotonic(), deepcopy(result))
                    self.entries.move_to_end(key)
                    while len(self.entries) > self.max_entries:
                        self.entries.popitem(last=False)
                    return result
                finally:
                    self.pending.pop(task_key, None)

            task = asyncio.create_task(fetch())
            self.pending[task_key] = task
        return deepcopy(await asyncio.shield(task))

    def clear(self) -> None:
        self.entries.clear()
