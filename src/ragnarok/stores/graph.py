"""Knowledge graph store (Step 34).

A lightweight entity graph built at ingest: nodes are entities (from enrichment, Step 6), edges are
co-occurrence relations (entities appearing in the same chunk), and each entity maps to the chunks
that mention it. Graph retrieval expands from the query's entities to their neighborhood and returns
the chunks that mention them — good for multi-hop / relationship questions the vector store answers
poorly.

In-memory by default (dev/CI); a Neo4j-backed variant scales it in prod (same interface).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any


@dataclass
class GraphHit:
    chunk_id: str
    payload: dict[str, Any]
    score: float
    matched_entities: list[str] = field(default_factory=list)


def _norm(entity: str) -> str:
    return entity.strip().lower()


class KnowledgeGraph:
    def __init__(self) -> None:
        self._edges: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._entity_chunks: dict[str, set[str]] = defaultdict(set)
        self._chunk_payload: dict[str, dict[str, Any]] = {}
        self._entities: set[str] = set()

    def add_chunk(self, chunk_id: str, payload: dict[str, Any]) -> None:
        entities = [_norm(e) for e in (payload.get("entities") or []) if e.strip()]
        self._chunk_payload[chunk_id] = payload
        for e in entities:
            self._entities.add(e)
            self._entity_chunks[e].add(chunk_id)
        for a, b in combinations(sorted(set(entities)), 2):  # co-occurrence edges
            self._edges[a][b] += 1
            self._edges[b][a] += 1

    def neighbors(self, entity: str, hops: int = 1) -> set[str]:
        entity = _norm(entity)
        frontier = {entity}
        seen = {entity}
        for _ in range(hops):
            nxt: set[str] = set()
            for e in frontier:
                nxt |= set(self._edges.get(e, {}))
            frontier = nxt - seen
            seen |= nxt
        return seen

    def match_entities(self, text: str) -> list[str]:
        """Entities from the graph that appear in the query text (substring match)."""
        t = text.lower()
        return [e for e in self._entities if e in t]

    def query(self, text: str, *, hops: int = 1, limit: int = 40) -> list[GraphHit]:
        seeds = self.match_entities(text)
        if not seeds:
            return []
        expanded: set[str] = set()
        for s in seeds:
            expanded |= self.neighbors(s, hops)

        # score each chunk by how many of the expanded entities it mentions
        chunk_scores: dict[str, tuple[float, list[str]]] = {}
        for e in expanded:
            weight = 2.0 if e in seeds else 1.0  # seed entities weigh more than neighbors
            for cid in self._entity_chunks.get(e, ()):
                prev, ents = chunk_scores.get(cid, (0.0, []))
                chunk_scores[cid] = (prev + weight, [*ents, e])
        hits = [
            GraphHit(cid, self._chunk_payload[cid], score, ents)
            for cid, (score, ents) in chunk_scores.items()
            if cid in self._chunk_payload
        ]
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]

    def stats(self) -> dict[str, int]:
        return {"entities": len(self._entities), "chunks": len(self._chunk_payload)}
