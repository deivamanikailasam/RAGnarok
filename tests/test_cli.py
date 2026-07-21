"""Smoke tests for the CLI and doctor health-check (Step 1)."""

from __future__ import annotations

from ragnarok.cli import build_parser, check_endpoint


def test_parser_builds_with_all_subcommands():
    parser = build_parser()
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"


def test_check_endpoint_unreachable_port():
    # A port nothing listens on must report FAIL, not raise.
    check = check_endpoint("nothing", "http://localhost:1")
    assert check.ok is False
    assert "unreachable" in check.detail


def test_ask_subcommand_parses_query():
    parser = build_parser()
    args = parser.parse_args(["ask", "what is the refund policy?"])
    assert args.query == "what is the refund policy?"
