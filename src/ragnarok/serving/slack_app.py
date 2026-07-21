"""Slack interface via Socket Mode (Step 23).

Socket Mode opens an OUTBOUND WebSocket to Slack, so no public inbound endpoint is required — the
right fit for a fully-local/firewalled deployment. Instant ack + streaming chat_update collapse
perceived latency to time-to-first-token; thread history feeds the query optimizer's coreference
(Step 14); feedback buttons feed post-prod eval (Step 25). slack_bolt is lazy-imported.
"""

from __future__ import annotations

from typing import Any

from ragnarok.user import User


def resolve_user(body: dict) -> User:
    """Map the Slack user to entitlements (via SSO group mapping in prod)."""
    return User(id=body.get("user", {}).get("id", "slack"), entitlements=["public"])


def _render(result: Any) -> str:
    text = result.text
    if result.citations:
        srcs = "\n".join(f"• <{c.uri}|{c.title} › {c.section}>" for c in result.citations)
        text += f"\n\n*Sources:*\n{srcs}"
    if result.abstained:
        text = ":warning: " + text
    return text


def build_slack_app() -> Any:  # pragma: no cover - requires slack_bolt + tokens
    import os

    from slack_bolt.app.async_app import AsyncApp

    from ragnarok.pipeline import answer
    from ragnarok.stores.factory import get_feature_store, get_vector_store

    app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])

    @app.event("app_mention")
    async def handle_mention(event: dict, say: Any, client: Any) -> None:
        user = resolve_user({"user": {"id": event.get("user")}})
        query = event.get("text", "")
        thread_ts = event.get("thread_ts") or event.get("ts")
        placeholder = await say(text=":mag: searching…", thread_ts=thread_ts)
        result = await answer(query, user, store=get_vector_store(), features=get_feature_store())
        await client.chat_update(
            channel=event["channel"], ts=placeholder["ts"], text=_render(result)
        )

    return app


def run_socket_mode() -> None:  # pragma: no cover - requires tokens
    import asyncio
    import os

    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

    handler = AsyncSocketModeHandler(build_slack_app(), os.environ["SLACK_APP_TOKEN"])
    asyncio.run(handler.start_async())
