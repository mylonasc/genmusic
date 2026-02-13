"""
System class for running the generative music framework.
"""

import asyncio
from typing import List

from .events import EventBus
from .nodes.base import Node


class System:
    """System that manages and runs all nodes."""

    def __init__(self, bus: EventBus = None):
        self.bus = bus or EventBus()
        self.nodes: List[Node] = []
        self._tasks: List[asyncio.Task] = []

    def add(self, node: Node) -> "System":
        """Add a node to the system."""
        self.nodes.append(node)
        return self

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)

    async def run_for(self, seconds: float) -> None:
        """Run all nodes for the specified duration."""
        self._tasks = [asyncio.create_task(n.start()) for n in self.nodes]
        try:
            await asyncio.sleep(max(0.0, seconds))
        finally:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def run_forever(self) -> None:
        """Run all nodes until cancelled."""
        self._tasks = [asyncio.create_task(n.start()) for n in self.nodes]
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for t in self._tasks:
                t.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)

    def stop(self) -> None:
        """Stop all nodes."""
        for node in self.nodes:
            node.stop()
        for t in self._tasks:
            t.cancel()
