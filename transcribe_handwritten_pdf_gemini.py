#!/usr/bin/env python3
"""
Chunked transcription pipeline for dense handwritten PDFs using Gemini.

Designed for low free-tier limits (e.g. 5 requests/minute, 20 requests/day):
1) Renders pages to PNG
2) Sends 1 request per chunk (multiple pages per call)
3) Enforces minute/day request limits
4) Saves per-page markdown, chunk responses, and combined transcription
5) Supports resume/checkpointing

Usage example:
  export GOOGLE_API_KEY="your_key_here"
  python transcribe_handwritten_pdf_gemini.py \
    --pdf "/Users/zafirnasim/Documents/ml1h/ScribbleTogether (2).pdf" \
    --out "/Users/zafirnasim/Documents/ml1h/scribble2_transcription" \
    --model "gemini-2.5-flash" \
    --chunk-size 3 \
    --max-requests-per-minute 5 \
    --max-requests-per-day 20 \
    --dpi 300
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import date
from typing import Optional

import fitz  # PyMuPDF
from google import genai
from google.genai import types


SYSTEM_PROMPT = """You are transcribing handwritten mathematical lecture notes from MULTIPLE pages in one request.

Primary goal: maximize faithful extraction of content.

Output format (STRICT, do this FOR EACH PAGE):

# Page {page_number}
## Literal transcription
- Capture all legible text in reading order.
- Preserve equations and symbols. Use LaTeX for math where possible.
- Keep bullets/numbering/structure from the page.
- If text is uncertain, mark with [?].

## Clean reconstruction
- Rewrite the page into coherent, readable notes without losing technical detail.
- Keep all formulas, assumptions, theorem/proof steps, and definitions.
- Do NOT invent new results not supported by the page.

## Ambiguities / low-confidence items
- List unclear tokens, symbols, or lines with best guess and confidence (low/med/high).

Be exhaustive. Do not summarize away details.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe handwritten PDF via Gemini (chunked requests).")
    parser.add_argument("--pdf", required=True, help="Absolute path to input PDF.")
    parser.add_argument("--out", required=True, help="Absolute path to output directory.")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model (vision-capable).")
    parser.add_argument("--dpi", type=int, default=300, help="Render DPI for each page image.")
    parser.add_argument("--chunk-size", type=int, default=3, help="Number of pages per Gemini request.")
    parser.add_argument("--max-retries", type=int, default=5, help="Retries per chunk on API failure.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Base delay after each successful request (seconds).")
    parser.add_argument("--max-requests-per-minute", type=int, default=5, help="Minute quota.")
    parser.add_argument("--max-requests-per-day", type=int, default=20, help="Daily quota.")
    parser.add_argument("--start-page", type=int, default=1, help="1-indexed start page.")
    parser.add_argument("--end-page", type=int, default=0, help="1-indexed end page (0 = last page).")
    parser.add_argument("--force", action="store_true", help="Reprocess pages even if output exists.")
    return parser.parse_args()


def require_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Missing GOOGLE_API_KEY in environment.")
    return key


def render_page_png_bytes(doc: fitz.Document, page_index: int, dpi: int) -> bytes:
    page = doc.load_page(page_index)
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")


def split_into_chunks(pages: list[int], chunk_size: int) -> list[list[int]]:
    return [pages[i : i + chunk_size] for i in range(0, len(pages), chunk_size)]


def parse_sections_by_page_number(response_text: str) -> dict[int, str]:
    """
    Parse sections like '# Page 12' and map page -> section text.
    """
    lines = response_text.splitlines()
    sections: dict[int, list[str]] = {}
    current_page: Optional[int] = None

    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("# page "):
            maybe_num = stripped[7:].strip()
            try:
                page_num = int(maybe_num.split()[0])
                current_page = page_num
                sections.setdefault(current_page, [])
                sections[current_page].append(line)
                continue
            except Exception:
                pass

        if current_page is not None:
            sections[current_page].append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def enforce_rate_limit(
    manifest: dict,
    max_requests_per_minute: int,
    max_requests_per_day: int,
) -> bool:
    """
    Return True if caller may proceed now, False if daily limit reached.
    Sleeps as needed for per-minute quota.
    """
    now = time.time()
    today = date.today().isoformat()
    req_state = manifest.setdefault("request_tracking", {})
    day_state = req_state.setdefault(today, {"count": 0, "timestamps": []})

    if day_state["count"] >= max_requests_per_day:
        return False

    # Keep only last 60s timestamps
    ts = [float(x) for x in day_state.get("timestamps", [])]
    ts = [x for x in ts if now - x < 60.0]
    day_state["timestamps"] = ts
    if len(ts) >= max_requests_per_minute:
        wait_for = 60.0 - (now - ts[0]) + 0.1
        if wait_for > 0:
            print(f"[rate] minute cap reached; sleeping {wait_for:.1f}s")
            time.sleep(wait_for)
    return True


def register_request(manifest: dict) -> None:
    today = date.today().isoformat()
    now = time.time()
    req_state = manifest.setdefault("request_tracking", {})
    day_state = req_state.setdefault(today, {"count": 0, "timestamps": []})
    day_state["count"] += 1
    day_state.setdefault("timestamps", []).append(now)


def call_gemini_with_retry(
    client: genai.Client,
    model: str,
    page_images: list[tuple[int, bytes]],
    max_retries: int,
) -> str:
    page_numbers = [p for p, _ in page_images]
    page_list = ", ".join(str(p) for p in page_numbers)
    prompt = (
        SYSTEM_PROMPT
        + "\n\nPages included in this request: "
        + page_list
        + "\nReturn one clearly separated section per page exactly as '# Page N'."
    )

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            parts = [types.Part.from_text(text=prompt)]
            for _, image_png in page_images:
                parts.append(types.Part.from_bytes(data=image_png, mime_type="image/png"))

            response = client.models.generate_content(
                model=model,
                contents=parts,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    top_p=0.95,
                ),
            )
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Empty response text from model.")
            return text
        except Exception as err:  # noqa: BLE001 - keep retries broad for API errors
            last_err = err
            backoff = min(60, 2 ** (attempt - 1))
            print(
                f"[warn] pages {page_numbers}: attempt {attempt}/{max_retries} failed: {err}",
                file=sys.stderr,
            )
            if attempt < max_retries:
                time.sleep(backoff)
            else:
                break

    raise RuntimeError(f"Failed pages {page_numbers} after {max_retries} attempts: {last_err}") from last_err


def main() -> int:
    args = parse_args()
    require_api_key()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    pages_dir = out_dir / "pages"
    images_dir = out_dir / "images"
    chunks_dir = out_dir / "chunks"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = genai.Client()
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)

    start = max(1, args.start_page)
    end = total_pages if args.end_page <= 0 else min(args.end_page, total_pages)
    if start > end:
        raise ValueError(f"Invalid range: start={start}, end={end}, total={total_pages}")

    manifest_path = out_dir / "manifest.json"
    manifest = {
        "pdf": str(pdf_path),
        "model": args.model,
        "dpi": args.dpi,
        "chunk_size": args.chunk_size,
        "total_pages": total_pages,
        "processed_pages": [],
        "failed_pages": [],
        "processed_chunks": [],
        "halt_reason": "",
        "generated_at_unix": int(time.time()),
    }
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    print(f"[info] PDF: {pdf_path}")
    print(f"[info] Pages: {total_pages} | range: {start}-{end}")
    print(f"[info] Output: {out_dir}")
    print(f"[info] Model: {args.model}")
    print(f"[info] Chunk size: {args.chunk_size}")
    print(f"[info] Quotas: {args.max_requests_per_minute}/min, {args.max_requests_per_day}/day")

    failed_pages: list[int] = []
    target_pages = list(range(start, end + 1))
    # Only process missing pages unless --force.
    if not args.force:
        done = set(int(x) for x in manifest.get("processed_pages", []))
        target_pages = [p for p in target_pages if p not in done]

    chunks = split_into_chunks(target_pages, max(1, args.chunk_size))
    for chunk_idx, chunk_pages in enumerate(chunks, start=1):
        if not chunk_pages:
            continue

        chunk_key = f"{chunk_pages[0]}-{chunk_pages[-1]}"
        print(f"[work] chunk {chunk_idx}/{len(chunks)} pages {chunk_key}")

        if not enforce_rate_limit(
            manifest=manifest,
            max_requests_per_minute=args.max_requests_per_minute,
            max_requests_per_day=args.max_requests_per_day,
        ):
            manifest["halt_reason"] = (
                f"daily request limit reached ({args.max_requests_per_day}); rerun tomorrow to continue"
            )
            print(f"[stop] {manifest['halt_reason']}")
            manifest["generated_at_unix"] = int(time.time())
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            break

        # Render images for this chunk.
        page_images: list[tuple[int, bytes]] = []
        for page_number in chunk_pages:
            page_idx = page_number - 1
            page_png = images_dir / f"page_{page_number:04d}.png"
            image_bytes = render_page_png_bytes(doc, page_idx, args.dpi)
            page_png.write_bytes(image_bytes)
            page_images.append((page_number, image_bytes))

        try:
            content = call_gemini_with_retry(
                client=client,
                model=args.model,
                page_images=page_images,
                max_retries=args.max_retries,
            )
            register_request(manifest)

            # Save raw chunk response.
            chunk_raw = chunks_dir / f"chunk_{chunk_pages[0]:04d}_{chunk_pages[-1]:04d}.md"
            chunk_raw.write_text(content + "\n", encoding="utf-8")

            # Parse response by page headers and save per-page files.
            parsed = parse_sections_by_page_number(content)
            for page_number in chunk_pages:
                page_md = pages_dir / f"page_{page_number:04d}.md"
                section = parsed.get(page_number)
                if section:
                    page_md.write_text(section + "\n", encoding="utf-8")
                    if page_number not in manifest.get("processed_pages", []):
                        manifest.setdefault("processed_pages", []).append(page_number)
                    print(f"[ok] page {page_number}")
                else:
                    # Fallback: store entire chunk output as trace and mark failed for re-run.
                    failed_pages.append(page_number)
                    manifest.setdefault("failed_pages", []).append(page_number)
                    fallback = (
                        f"# Page {page_number}\n\n"
                        "## Parsing warning\n"
                        "Model response did not include a clean '# Page N' section for this page.\n\n"
                        "## Raw chunk response\n"
                        + content
                    )
                    page_md.write_text(fallback + "\n", encoding="utf-8")
                    print(f"[warn] page {page_number}: missing explicit section; wrote fallback", file=sys.stderr)

            manifest.setdefault("processed_chunks", []).append(chunk_key)
        except Exception as err:  # noqa: BLE001
            register_request(manifest)
            failed_pages.extend(chunk_pages)
            for p in chunk_pages:
                manifest.setdefault("failed_pages", []).append(p)
            print(f"[fail] chunk {chunk_key}: {err}", file=sys.stderr)

        manifest["generated_at_unix"] = int(time.time())
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        time.sleep(args.sleep)

    # Build combined markdown in page order from existing per-page files.
    combined_path = out_dir / "combined_transcription.md"
    with combined_path.open("w", encoding="utf-8") as f:
        f.write(f"# Combined Transcription\n\nSource: `{pdf_path}`\n\n")
        for page_number in range(1, total_pages + 1):
            page_md = pages_dir / f"page_{page_number:04d}.md"
            if page_md.exists():
                f.write(page_md.read_text(encoding="utf-8").rstrip() + "\n\n---\n\n")

    print(f"[done] Combined output: {combined_path}")
    if failed_pages:
        print(f"[warn] Failed pages: {failed_pages}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
