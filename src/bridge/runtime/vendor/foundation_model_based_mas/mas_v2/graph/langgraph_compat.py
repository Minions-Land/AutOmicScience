from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


START = "__start__"
END = "__end__"


@dataclass
class _CompiledGraph:
    nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]]
    edges: dict[str, str]
    conditional_edges: dict[str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]] = field(default_factory=dict)

    def invoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        current = self.edges.get(START, END)
        current_state = dict(state)
        while current != END:
            update = self.nodes[current](current_state) or {}
            current_state.update(update)
            if current in self.conditional_edges:
                router, mapping = self.conditional_edges[current]
                route_name = router(current_state)
                current = mapping[route_name]
            else:
                current = self.edges[current]
        return current_state


class StateGraph:
    def __init__(self, _state_type: Any) -> None:
        self.nodes: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}
        self.edges: dict[str, str] = {}
        self.conditional_edges: dict[str, tuple[Callable[[dict[str, Any]], str], dict[str, str]]] = {}

    def add_node(self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self.nodes[name] = fn

    def add_edge(self, source: str, target: str) -> None:
        self.edges[source] = target

    def add_conditional_edges(
        self,
        source: str,
        router: Callable[[dict[str, Any]], str],
        mapping: dict[str, str],
    ) -> None:
        self.conditional_edges[source] = (router, mapping)

    def compile(self) -> _CompiledGraph:
        return _CompiledGraph(nodes=dict(self.nodes), edges=dict(self.edges), conditional_edges=dict(self.conditional_edges))
