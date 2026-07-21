"""Build the retrieval filter with access control enforced as data (Step 15).

The source identifier (Step 14) may *narrow* which sources to search, but entitlements are the hard
boundary: access_tags on the filter are always the intersection of the model's suggestion with the
caller's real entitlements (falling back to entitlements alone). So an unauthorized chunk is never
scored — security by construction, not by prompt instruction.
"""

from __future__ import annotations

from ragnarok.retrieval.preprocess import SourcePlan
from ragnarok.stores.vector import MetadataFilter
from ragnarok.user import User


def build_filter(
    source: SourcePlan | None, user: User, *, freshness_after: str | None = None
) -> MetadataFilter:
    equals = dict(source.equals) if source else {}
    suggested = source.must_access_tags if source else []
    # SECURITY: model-proposed tags are intersected with real entitlements; never trusted to widen.
    scoped = [t for t in suggested if t in user.entitlements] or list(user.entitlements)
    return MetadataFilter(
        equals=equals,
        access_tags=scoped,
        freshness_after=freshness_after,
    )


def relax_filter(user: User) -> MetadataFilter:
    """Access-scoped but otherwise unfiltered — the fallback when a narrow filter starves retrieval.

    Crucially still bounded by entitlements, so relaxing metadata filters never relaxes security.
    """
    return MetadataFilter(access_tags=list(user.entitlements))
