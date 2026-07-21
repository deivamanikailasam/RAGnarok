"""Content enrichment (Step 6) — the biggest quality lever before retrieval.

Runs the large LLM once per document (offline, so per-query cost is unaffected) to produce:
- table descriptions  -> turn a pricing/SLA matrix into sentences that semantic search can match
- doc/section summaries -> contextual prefixes (Step 8) + source routing (Step 14) + display
- custom metadata      -> doc_type, audience/tier, topics, entities, freshness, authority,
                          access_tags: powers metadata-filtered retrieval + recency/authority
                          ranking + security (Steps 14-18, 22)

Structured (guided-JSON) output keeps the metadata reliable and cheap. A fallback to llm_small
keeps enrichment resilient (labelled degraded upstream).
"""

from __future__ import annotations

import asyncio
from typing import Literal, Optional

from pydantic import BaseModel, Field

from ragnarok.config import get_settings
from ragnarok.ingestion.models import NormalizedDoc, Section
from ragnarok.prompts import prompts
from ragnarok.structured import generate_structured


class SectionSummary(BaseModel):
    heading: str
    summary: str


class TableDesc(BaseModel):
    section_heading: str
    description: str


class CustomMetadata(BaseModel):
    topics: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    doc_type: Literal["policy", "runbook", "spec", "faq", "contract", "report", "other"] = "other"
    audience: list[str] = Field(default_factory=list)  # e.g. ["enterprise","consumer"]
    freshness_date: Optional[str] = None  # ISO date if the doc states an effective/updated date
    authority: Literal["official", "draft", "deprecated"] = "draft"
    access_tags: list[str] = Field(default_factory=list)


class Enrichment(BaseModel):
    doc_summary: str = ""
    section_summaries: list[SectionSummary] = Field(default_factory=list)
    table_descriptions: list[TableDesc] = Field(default_factory=list)
    custom_metadata: CustomMetadata = Field(default_factory=CustomMetadata)
    keywords: list[str] = Field(default_factory=list)


class EnrichedDocument(BaseModel):
    doc_id: str
    uri: str
    source_type: str
    title: str
    sections: list[Section]
    acl_tags: list[str]
    content_hash: str
    enrichment: Enrichment
    enrichment_model: str = ""
    enrichment_version: str = ""

    def summary_for(self, heading: str) -> str:
        for s in self.enrichment.section_summaries:
            if s.heading == heading:
                return s.summary
        return self.enrichment.doc_summary

    def table_desc_for(self, heading: str) -> str:
        for t in self.enrichment.table_descriptions:
            if t.section_heading == heading:
                return t.description
        return ""

    @classmethod
    def from_norm(
        cls, norm: NormalizedDoc, enr: Enrichment, *, model: str, version: str
    ) -> "EnrichedDocument":
        # ACLs from the source become access_tags if the LLM didn't infer them (security default).
        if not enr.custom_metadata.access_tags:
            enr.custom_metadata.access_tags = list(norm.acl_tags)
        return cls(
            doc_id=norm.doc_id,
            uri=norm.uri,
            source_type=norm.source_type,
            title=norm.title,
            sections=norm.sections,
            acl_tags=norm.acl_tags,
            content_hash=norm.content_hash,
            enrichment=enr,
            enrichment_model=model,
            enrichment_version=version,
        )


async def enrich(norm: NormalizedDoc, *, role_name: str = "llm_large") -> EnrichedDocument:
    version = prompts().label("content_enricher")
    # Enrichment cache (Step 7): key on content hash + prompt version, so re-ingest of unchanged
    # content is a cache hit and a prompt/model bump re-enriches only affected docs.
    from ragnarok.cache import get_cache, get_json, make_key, set_json

    cache = get_cache()
    key = make_key("enr", version, norm.content_hash)
    if (cached := get_json(cache, key)) is not None:
        return EnrichedDocument.model_validate(cached)

    msgs = prompts().render(
        "content_enricher", "latest", title=norm.title, content=norm.render_for_llm()
    )
    enr = await generate_structured(role_name, msgs, Enrichment, fallback_role="llm_small")
    model = getattr(get_settings().models, role_name).model
    doc = EnrichedDocument.from_norm(norm, enr, model=model, version=version)
    set_json(cache, key, doc.model_dump())
    return doc


def enrich_sync(norm: NormalizedDoc) -> EnrichedDocument:
    """Sync wrapper for the (sync) ingestion pipeline. Safe outside an event loop."""
    return asyncio.run(enrich(norm))
