"""Fetch closed GitHub issues and PRs as Chunks.

REST API, anonymous-by-default (60 req/hr). One request per page of 100 items.
Each item becomes one chunk with synthetic source path `_gh/{repo}/{issues|pr}/{N}`.
URLs go in chunk metadata so the CLI can render clickable citations."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterator

import httpx

from .ingest import Chunk

GH_API = "https://api.github.com"
BODY_MAX = 6000  # chars — keeps issue chunks roughly comparable in size to code chunks
PAGE_PAUSE_SEC = 0.5  # be polite between page requests


@dataclass
class GHItem:
    number: int
    is_pr: bool
    title: str
    body: str
    state: str
    labels: list[str]
    url: str
    closed_at: str | None


def fetch_items(owner: str, repo: str, max_items: int, token: str | None = None) -> Iterator[GHItem]:
    """Yield up to `max_items` closed issues + PRs, most-recently-updated first."""
    token = token or os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "eih/0.2"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    yielded = 0
    page = 1
    with httpx.Client(headers=headers, timeout=30.0) as client:
        while yielded < max_items:
            resp = client.get(
                f"{GH_API}/repos/{owner}/{repo}/issues",
                params={
                    "state": "closed", "sort": "updated", "direction": "desc",
                    "per_page": 100, "page": page,
                },
            )
            if resp.status_code == 403 and "rate limit" in resp.text.lower():
                reset = int(resp.headers.get("x-ratelimit-reset", 0))
                wait = max(0, reset - int(time.time())) + 5
                print(f"  rate-limited; sleeping {wait}s")
                time.sleep(wait)
                continue
            if resp.status_code == 422:
                # GitHub caps deep pagination at page 10 (1000 results) on /issues.
                # Stop gracefully — caller keeps whatever was yielded so far.
                print(f"  GitHub deep-pagination cap hit at page {page} (~{yielded} items collected). Stopping.")
                return
            resp.raise_for_status()
            items = resp.json()
            if not items:
                return
            for raw in items:
                if yielded >= max_items:
                    return
                yield GHItem(
                    number=raw["number"],
                    is_pr="pull_request" in raw,
                    title=raw.get("title") or "",
                    body=raw.get("body") or "",
                    state=raw.get("state") or "closed",
                    labels=[l["name"] for l in (raw.get("labels") or [])],
                    url=raw.get("html_url") or "",
                    closed_at=raw.get("closed_at"),
                )
                yielded += 1
            page += 1
            time.sleep(PAGE_PAUSE_SEC)


def item_to_chunk(item: GHItem, repo: str) -> Chunk | None:
    title = item.title.strip()
    if not title:
        return None
    body = (item.body or "").strip()
    if len(body) > BODY_MAX:
        body = body[:BODY_MAX] + "\n\n[... truncated ...]"
    kind = "pr" if item.is_pr else "issue"
    labels_str = ", ".join(item.labels) if item.labels else "none"
    parts = [
        f"# {title}",
        f"#{item.number} · {kind.upper()} · state: {item.state} · labels: {labels_str}",
    ]
    if body:
        parts.append(body)
    text = "\n\n".join(parts)
    path_segment = "pr" if item.is_pr else "issues"
    return Chunk(
        text=text,
        source_path=f"_gh/{repo}/{path_segment}/{item.number}",
        kind=kind,
        symbol=title[:80],
        start_line=1,
        end_line=text.count("\n") + 1,
    )
