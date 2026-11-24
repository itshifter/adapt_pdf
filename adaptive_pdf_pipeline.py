import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from pydantic import BaseModel, Field
from ollama import Client, ResponseError


# ---------------------------------------------------------
# 1. Target Schema & Multi-Page Continuation State
# ---------------------------------------------------------
class ExtractedRow(BaseModel):
    item_code: Optional[str] = Field(None, description="SKU, part number, or line ID")
    description: str = Field(..., description="Clean item or line description")
    quantity: Optional[float] = Field(None, description="Numeric quantity")
    unit_price: Optional[float] = Field(None, description="Unit price as float")
    total_amount: Optional[float] = Field(None, description="Total line amount as float")
    category: Optional[str] = Field(None, description="Active section header or group")
    llm_confidence: float = Field(
        ...,
        description="Confidence between 0.0 and 1.0 that cell alignment and values are unambiguous",
    )
    ambiguity_note: Optional[str] = Field(
        None,
        description="Briefly note any merged, clipped, or unclear cell values; null if clear",
    )


class PageExtraction(BaseModel):
    document_entity: Optional[str] = Field(None, description="Vendor, issuer, or account name")
    document_date: Optional[str] = Field(None, description="Document date in YYYY-MM-DD format")
    detected_headers: list[str] = Field(
        default_factory=list,
        description="Column headers active at the bottom of this page for multi-page continuation",
    )
    rows: list[ExtractedRow] = Field(default_factory=list)


class ContinuationState(BaseModel):
    document_entity: Optional[str] = None
    document_date: Optional[str] = None
    active_category: Optional[str] = None
    active_headers: list[str] = Field(default_factory=list)
    tail_rows: list[dict] = Field(default_factory=list)


# ---------------------------------------------------------
# 2. Stage 1: Isolated Docling Parser (Frees 100% RAM on Exit)
# ---------------------------------------------------------
def run_docling_stage(pdf_path: Path, cache_json: Path) -> None:
    """
    Imports Docling/PyTorch only inside this child process, parses the PDF
    with adaptive OCR fallback, saves structured page data to `cache_json`,
    and exits so Linux reclaims all PyTorch/ONNX memory before Ollama runs.
    """
    import gc
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        RapidOcrOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    def build_converter(force_ocr: bool = False) -> DocumentConverter:
        opts = PdfPipelineOptions()
        opts.do_ocr = True
        opts.do_table_structure = True
        opts.table_structure_options.mode = TableFormerMode.ACCURATE
        opts.table_structure_options.do_cell_matching = True
        opts.ocr_options = RapidOcrOptions(force_full_page_ocr=force_ocr)
        opts.images_scale = 1.5 if force_ocr else 1.0
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )

    fast_conv = build_converter(force_ocr=False)
    result = fast_conv.convert(pdf_path)
    del fast_conv
    gc.collect()

    conf = getattr(result, "confidence", None)
    low_grade = str(getattr(conf, "low_grade", "GOOD")).upper()
    raw_fast_mean = getattr(conf, "mean_score", None)
    fast_mean = (
        raw_fast_mean
        if isinstance(raw_fast_mean, (int, float)) and math.isfinite(raw_fast_mean)
        else 0.0
    )
    parse_mode = "hybrid_native"

    if any(bad in low_grade for bad in ("POOR", "FAIR")):
        print(f"[Fallback] Low initial grade ({low_grade}) on {pdf_path.name}; running full-page OCR...")
        ocr_conv = build_converter(force_ocr=True)
        ocr_result = ocr_conv.convert(pdf_path)
        del ocr_conv
        gc.collect()

        ocr_conf = getattr(ocr_result, "confidence", None)
        raw_ocr_mean = getattr(ocr_conf, "mean_score", None)
        ocr_mean = (
            raw_ocr_mean
            if isinstance(raw_ocr_mean, (int, float)) and math.isfinite(raw_ocr_mean)
            else 0.0
        )
        if ocr_mean >= fast_mean:
            result = ocr_result
            parse_mode = "full_ocr_fallback"

    doc = result.document
    conf_report = getattr(result, "confidence", None)
    page_conf_map = getattr(conf_report, "pages", {}) or {}

    pages_payload = []
    for page_no in sorted(doc.pages.keys()):
        page_md = doc.export_to_markdown(page_no=page_no)

        html_blocks: list[str] = []
        merged_cell_count = 0
        for idx, table in enumerate(doc.tables, start=1):
            table_pages = {p.page_no for p in getattr(table, "prov", [])}
            if page_no not in table_pages:
                continue
            if getattr(table, "data", None) and getattr(table.data, "table_cells", None):
                for cell in table.data.table_cells:
                    if getattr(cell, "row_span", 1) > 1 or getattr(cell, "col_span", 1) > 1:
                        merged_cell_count += 1
            html_blocks.append(f"<!-- Table {idx} -->\n{table.export_to_html(doc=doc)}")

        page_conf = page_conf_map.get(page_no)
        doc_signals = [
            float(s)
            for s in (
                getattr(page_conf, "parse_score", None),
                getattr(page_conf, "ocr_score", None),
                getattr(page_conf, "layout_score", None),
            )
            if isinstance(s, (int, float)) and math.isfinite(s)
        ]
        docling_score = round(sum(doc_signals) / len(doc_signals), 4) if doc_signals else 0.90
        page_grade = str(getattr(page_conf, "low_grade", "UNKNOWN"))

        pages_payload.append(
            {
                "page_no": page_no,
                "page_md": page_md,
                "tables_html": "\n\n".join(html_blocks),
                "merged_count": merged_cell_count,
                "docling_score": docling_score,
                "page_grade": page_grade,
            }
        )

    cache_json.parent.mkdir(parents=True, exist_ok=True)
    cache_json.write_text(
        json.dumps(
            {
                "parse_mode": parse_mode,
                "total_pages": len(doc.pages),
                "pages": pages_payload,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------
# 3. 3-Layer Confidence Scoring & Human-Review Flagging
# ---------------------------------------------------------
def score_and_flag_row(
    row: ExtractedRow,
    docling_score: float,
    merged_cell_count: int,
    prev_tail_descriptions: set[str],
) -> tuple[float, list[str]]:
    flags: list[str] = []

    if not math.isfinite(docling_score):
        docling_score = 0.85

    llm_conf = max(0.0, min(1.0, row.llm_confidence))

    if docling_score < 0.75:
        flags.append(f"LOW_DOCLING_PAGE_QUALITY ({docling_score:.2f})")

    structure_score = 1.0
    if merged_cell_count > 0:
        structure_score -= min(0.15, merged_cell_count * 0.02)
    if row.ambiguity_note:
        flags.append(f"LLM_AMBIGUITY: {row.ambiguity_note}")
        structure_score -= 0.15

    math_score = 1.0
    if row.quantity is not None and row.unit_price is not None and row.total_amount is not None:
        expected = round(row.quantity * row.unit_price, 2)
        if abs(expected - row.total_amount) > 0.02:
            flags.append(
                f"MATH_MISMATCH ({row.quantity} * {row.unit_price} = {expected}, got {row.total_amount})"
            )
            math_score -= 0.35

    if row.total_amount is None and row.unit_price is None:
        flags.append("MISSING_MONETARY_VALUES")
        math_score -= 0.25

    if len(row.description.strip()) < 3:
        flags.append("SUSPICIOUSLY_SHORT_DESCRIPTION")
        math_score -= 0.25

    if row.description.strip().lower() in prev_tail_descriptions:
        flags.append("POSSIBLE_CROSS_PAGE_DUPLICATE")
        math_score -= 0.20

    composite = (
        0.35 * docling_score
        + 0.35 * llm_conf
        + 0.15 * structure_score
        + 0.15 * math_score
    )
    return round(max(0.0, min(1.0, composite)), 3), flags


# ---------------------------------------------------------
# 4. Stage 2: Streaming AI Extraction & Export
# ---------------------------------------------------------
def stream_structured_chat(client: Client, model_name: str, prompt: str) -> PageExtraction:
    """Streams structured JSON tokens so CPU inference never hits ReadTimeout."""
    chunks: list[str] = []
    token_count = 0

    stream = client.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        format=PageExtraction.model_json_schema(),
        options={"temperature": 0, "num_ctx": 2048},
        stream=True,
    )

    for chunk in stream:
        piece = chunk.message.content or ""
        if piece:
            chunks.append(piece)
            token_count += 1
            if token_count % 25 == 0:
                print(".", end="", flush=True)

    print()  # Newline after progress dots
    return PageExtraction.model_validate_json("".join(chunks))


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    model_name: str = "qwen2.5:3b",
    review_threshold: float = 0.85,
    force_reparse: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_json = output_dir / f".{pdf_path.stem}_parsed.json"

    # Run Docling in an isolated subprocess (or reuse fresh cache)
    if force_reparse or not cache_json.exists() or cache_json.stat().st_mtime < pdf_path.stat().st_mtime:
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--parse-only", str(pdf_path), str(cache_json)],
            check=True,
        )
    else:
        print(f"[Cache] Using existing parsed layout: {cache_json}")

    parsed_doc = json.loads(cache_json.read_text(encoding="utf-8"))
    parse_mode = parsed_doc["parse_mode"]
    total_pages = parsed_doc["total_pages"]

    client = Client(host="http://127.0.0.1:11434", timeout=600.0)
    try:
        client.show(model_name)
    except ResponseError:
        raise RuntimeError(f"Model '{model_name}' is not installed. Run: ollama pull {model_name}")
    except Exception as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at 127.0.0.1:11434 ({exc}). Start it with: ollama serve"
        )

    state = ContinuationState()
    verified_records: list[dict] = []
    review_records: list[dict] = []

    for page in parsed_doc["pages"]:
        page_no = page["page_no"]
        page_md = page["page_md"]
        tables_html = page["tables_html"]
        merged_count = page["merged_count"]
        docling_score = page["docling_score"]
        page_grade = page["page_grade"]

        if not page_md.strip() and not tables_html.strip():
            continue

        prompt = (
            "Extract all structured table rows from this PDF page into JSON.\n"
            "Rules:\n"
            "- Expand merged HTML cells (`rowspan`/`colspan`) onto every row they cover.\n"
            "- If a table continues from the previous page without repeating headers "
            "(including headerless rows in PROSE when TABLES is empty), use CONTINUATION STATE.\n"
            "- Strip currency symbols and commas from numbers; use null if missing.\n"
            "- Set `llm_confidence` (0.0-1.0) below 0.80 and fill `ambiguity_note` if uncertain.\n\n"
            f"CONTINUATION STATE:\n{state.model_dump_json()}\n\n"
            f"PAGE {page_no}/{total_pages} PROSE:\n{page_md}\n\n"
            f"PAGE {page_no} TABLES (HTML):\n"
            f"{tables_html or 'No explicit table grid; extract rows from PROSE above.'}"
        )

        print(f"[AI] Extracting page {page_no}/{total_pages} with {model_name} ", end="", flush=True)
        extracted = stream_structured_chat(client, model_name, prompt)

        current_entity = extracted.document_entity or state.document_entity
        current_date = extracted.document_date or state.document_date
        prev_tail_descs = {r.get("description", "").strip().lower() for r in state.tail_rows}

        for row in extracted.rows:
            if not row.category and state.active_category:
                row.category = state.active_category

            score, flags = score_and_flag_row(row, docling_score, merged_count, prev_tail_descs)

            record = row.model_dump()
            record.update(
                {
                    "document_entity": current_entity,
                    "document_date": current_date,
                    "source_file": pdf_path.name,
                    "page_number": page_no,
                    "parse_mode": parse_mode,
                    "docling_page_grade": page_grade,
                    "confidence_score": score,
                    "review_flags": " | ".join(flags) if flags else "",
                }
            )

            if score >= review_threshold and not flags:
                verified_records.append(record)
            else:
                review_records.append(record)

        state.document_entity = current_entity
        state.document_date = current_date
        if extracted.detected_headers:
            state.active_headers = extracted.detected_headers
        if extracted.rows:
            state.active_category = extracted.rows[-1].category or state.active_category
            state.tail_rows = [r.model_dump() for r in extracted.rows[-2:]]

    verified_df = pd.DataFrame(verified_records)
    review_df = pd.DataFrame(review_records)

    if not verified_df.empty:
        verified_df.to_excel(output_dir / f"{pdf_path.stem}_verified.xlsx", index=False)
    if not review_df.empty:
        review_df.to_excel(output_dir / f"{pdf_path.stem}_needs_review.xlsx", index=False)

    print(
        f"Completed {pdf_path.name} [{parse_mode}]: "
        f"{len(verified_df)} auto-verified rows | {len(review_df)} flagged for human review"
    )
    return verified_df, review_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parse-only", nargs=2, metavar=("PDF_PATH", "CACHE_JSON"))
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--reparse", action="store_true", help="Force re-running Docling OCR stage")
    args = parser.parse_args()

    if args.parse_only:
        run_docling_stage(Path(args.parse_only[0]), Path(args.parse_only[1]))
    else:
        process_pdf(
            Path("sample_document.pdf"),
            Path("output"),
            model_name=args.model,
            force_reparse=args.reparse,
        )