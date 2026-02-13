"""
Base node class for genmusic.
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .events import EventBus


class Node(ABC):
    """
    Base class for all nodes in the system.

    Nodes are async components that:
    - Subscribe to events from the EventBus
    - Process events and optionally publish new events
    - Run concurrently with other nodes
    """

    @abstractmethod
    async def start(self) -> None:
        """Start the node. Should run until cancelled."""
        pass

    def stop(self) -> None:
        """Stop the node. Override for cleanup."""
        pass


class NodeGroup:
    """Group of nodes that can be started/stopped together."""

    def __init__(self):
        self.nodes: List[Node] = []

    def add(self, node: Node) -> "NodeGroup":
        """Add a node to the group."""
        self.nodes.append(node)
        return self

    def __iter__(self):
        return iter(self.nodes)

    def __len__(self):
        return len(self.nodes)
