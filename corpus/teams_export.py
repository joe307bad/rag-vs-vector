"""
Export a Microsoft Teams channel's full history into the corpus: one Markdown file per thread.

Graph's channel-messages API needs admin consent, so this drives the Teams web client instead: it signs in
through Edge (Windows SSO, or you sign in in the window it opens), takes the headers the client itself sends
to the Teams chat service, and pages back through the channel to its first message. Nothing it can't
already show you in Teams.

    python -m corpus.teams_export fetch "<channel link>" --raw cache/teams/<name>
    python -m corpus.teams_export convert cache/teams/<name> --out <corpus folder>

`fetch` writes the raw message pages (JSON) and is the only step that touches Teams. `convert` is offline and
can be re-run. Needs `pip install playwright` and Microsoft Edge.
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

# Not config.CACHE_DIR: importing config exits when manifest.yaml is missing, and fetch doesn't need it.
PROFILE_DIR = Path(__file__).resolve().parent.parent / "cache" / "teams-browser-profile"
LOGIN_TIMEOUT_S = 180


# ── fetch ────────────────────────────────────────────────────────────────────

def _parse_channel_link(link: str) -> tuple[str, str]:
    """(channel thread id, channel name) from a 'Get link to channel' URL."""
    parts = urlparse(link).path.split("/")
    i = parts.index("channel")
    return unquote(parts[i + 1]), unquote(parts[i + 2]) if len(parts) > i + 2 else ""


def fetch(link: str, raw_dir: Path) -> None:
    from playwright.sync_api import sync_playwright

    channel_id, channel_name = _parse_channel_link(link)
    raw_dir.mkdir(parents=True, exist_ok=True)
    captured: dict = {}

    def on_request(req):
        # The client's own chat-service calls carry the auth we reuse. Kept in memory only.
        if "chatsvc" not in captured and "/api/chatsvc/" in req.url and "/users/ME/conversations/" in req.url:
            headers = {k: v for k, v in req.headers.items() if k.lower() not in ("content-length", "host")}
            if "authorization" in {k.lower() for k in headers}:
                captured["base"] = req.url.split("/users/ME/")[0]
                captured["chatsvc"] = headers

    with sync_playwright() as p:
        ctx = p.chromium.launch_persistent_context(
            str(PROFILE_DIR), channel="msedge", headless=False, viewport={"width": 1400, "height": 900}
        )
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.on("request", on_request)
        page.goto("https://teams.microsoft.com/v2/")
        print("Waiting for Teams to sign in (sign in in the Edge window if asked)...", flush=True)
        deadline = time.time() + LOGIN_TIMEOUT_S
        while "chatsvc" not in captured:
            if time.time() > deadline:
                ctx.close()
                sys.exit(f"Teams did not finish signing in within {LOGIN_TIMEOUT_S}s.")
            page.wait_for_timeout(2000)

        base, headers = captured["base"], captured["chatsvc"]
        url = (f"{base}/users/ME/conversations/{quote(channel_id, safe='')}/messages"
               "?view=msnp24Equivalent|supportsMessageProperties&pageSize=200&startTime=1")
        n_pages = n_msgs = 0
        while url:
            resp = ctx.request.get(url, headers=headers)
            if resp.status == 401:
                sys.exit("Teams token expired mid-export; re-run fetch (pages already written are kept).")
            if not resp.ok:
                sys.exit(f"Chat service returned {resp.status}: {resp.text()[:300]}")
            body = resp.json()
            msgs = body.get("messages", [])
            (raw_dir / f"page_{n_pages:04d}.json").write_text(json.dumps(msgs, ensure_ascii=False), encoding="utf-8")
            n_pages += 1
            n_msgs += len(msgs)
            oldest = min((m.get("originalarrivaltime", "") for m in msgs), default="")
            print(f"  page {n_pages}: {len(msgs)} messages (back to {oldest[:10]}), {n_msgs} total", flush=True)
            url = body.get("_metadata", {}).get("backwardLink") if msgs else None
        ctx.close()

    (raw_dir / "channel.json").write_text(
        json.dumps({"id": channel_id, "name": channel_name, "link": link, "fetched": datetime.now().isoformat()}),
        encoding="utf-8",
    )
    print(f"Fetched {n_msgs} messages in {n_pages} pages into {raw_dir}")


# ── convert ──────────────────────────────────────────────────────────────────

class _HtmlToMarkdown(HTMLParser):
    """Just enough HTML → Markdown for Teams messages: paragraphs, breaks, links, code, lists, quotes."""

    BLOCK = {"p", "div", "br", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "table"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.href: list[str | None] = []
        self.in_pre = 0
        self.in_code = 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.BLOCK:
            self.out.append("\n")
        if tag == "li":
            self.out.append("\n- ")
        elif tag == "pre":
            self.in_pre += 1
            self.out.append("\n```\n")
        elif tag == "code" and not self.in_pre:
            self.in_code += 1
            self.out.append("`")
        elif tag == "a":
            self.href.append(a.get("href"))
            self.out.append("[")
        elif tag == "img":
            self.out.append(f"[image: {a.get('alt') or 'attachment'}]")
        elif tag in ("b", "strong"):
            self.out.append("**")
        elif tag == "td":
            self.out.append(" | ")

    def handle_endtag(self, tag):
        if tag == "pre":
            self.in_pre -= 1
            self.out.append("\n```\n")
        elif tag == "code" and not self.in_pre and self.in_code:
            self.in_code -= 1
            self.out.append("`")
        elif tag == "a":
            href = self.href.pop() if self.href else None
            self.out.append(f"]({href})" if href else "]")
        elif tag in ("b", "strong"):
            self.out.append("**")
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        self.out.append(data if self.in_pre else re.sub(r"[ \t\r\n]+", " ", data))

    def text(self) -> str:
        s = "".join(self.out).replace("\xa0", " ")
        s = re.sub(r"\[([^\]]*)\]\(\1\)", r"<\1>", s)  # link whose text is its URL
        s = re.sub(r"[ \t]+\n", "\n", s)
        return re.sub(r"\n{3,}", "\n\n", s).strip()


def _to_markdown(content: str, message_type: str) -> str:
    if "html" not in message_type.lower():
        return html.unescape(content).strip()
    parser = _HtmlToMarkdown()
    parser.feed(content)
    return parser.text()


def _root_id(m: dict) -> str:
    if m.get("rootMessageId"):
        return str(m["rootMessageId"])
    match = re.search(r";messageid=(\d+)", m.get("conversationLink", "") + m.get("conversationid", ""))
    return match.group(1) if match else m["id"]


def _slug(text: str, limit: int = 60) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:limit].rstrip("-") or "thread"


def _subject(props) -> str:
    if isinstance(props, dict):
        return (props.get("subject") or "").strip()
    return ""


def _is_content(m: dict) -> bool:
    return (
        m.get("messagetype", "").lower() in ("richtext/html", "text", "richtext")
        and not (m.get("properties") or {}).get("deletetime")
        and bool(m.get("content", "").strip())
    )


def convert(raw_dir: Path, out_dir: Path) -> None:
    channel = json.loads((raw_dir / "channel.json").read_text(encoding="utf-8"))
    by_id: dict[str, dict] = {}
    for page in sorted(raw_dir.glob("page_*.json")):
        for m in json.loads(page.read_text(encoding="utf-8")):
            prev = by_id.get(m["id"])
            if prev is None or int(m.get("version") or 0) >= int(prev.get("version") or 0):
                by_id[m["id"]] = m  # keep the latest edit

    threads: dict[str, list[dict]] = defaultdict(list)
    for m in by_id.values():
        if _is_content(m):
            threads[_root_id(m)].append(m)

    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for root_id, msgs in threads.items():
        msgs.sort(key=lambda m: m.get("originalarrivaltime", ""))
        root = next((m for m in msgs if m["id"] == root_id), msgs[0])
        body_of = {m["id"]: _to_markdown(m["content"], m.get("messagetype", "")) for m in msgs}
        subject = _subject(root.get("properties")) or body_of[root["id"]].split("\n", 1)[0][:100] or "Untitled thread"
        started = root.get("originalarrivaltime", "")[:10]
        author = root.get("imdisplayname") or root.get("fromDisplayNameInToken") or "unknown"

        lines = [
            f"# {subject}",
            "",
            f"_Teams channel: {channel['name']} · started {started} by {author} · "
            f"{len(msgs) - 1} {'reply' if len(msgs) == 2 else 'replies'}_",
            "",
            "## Original post",
            "",
            body_of[root["id"]],
        ]
        replies = [m for m in msgs if m is not root]
        if replies:
            lines += ["", "## Replies", ""]
            for m in replies:
                who = m.get("imdisplayname") or m.get("fromDisplayNameInToken") or "unknown"
                when = m.get("originalarrivaltime", "")[:16].replace("T", " ")
                lines += [f"**{who}** ({when}):", "", body_of[m["id"]], ""]

        name = f"{started}-{_slug(subject)}-{root_id[-6:]}.md"
        (out_dir / name).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        written += 1
    print(f"Wrote {written} threads ({sum(len(v) for v in threads.values())} messages) to {out_dir}")


def main():
    ap = argparse.ArgumentParser(prog="python -m corpus.teams_export", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="Download the channel's full history as raw JSON pages")
    f.add_argument("link", help="Channel link from Teams (… > Get link to channel)")
    f.add_argument("--raw", required=True, type=Path, help="Folder for the raw pages")
    c = sub.add_parser("convert", help="Turn raw pages into one Markdown file per thread")
    c.add_argument("raw", type=Path)
    c.add_argument("--out", required=True, type=Path, help="Corpus folder to write Markdown into")
    args = ap.parse_args()
    if args.cmd == "fetch":
        fetch(args.link, args.raw)
    else:
        convert(args.raw, args.out)


if __name__ == "__main__":
    main()
