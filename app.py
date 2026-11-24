import shutil
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from adaptive_pdf_pipeline import process_pdf

app = FastAPI(title="Adaptive PDF Extractor")

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def df_to_clean_records(df: pd.DataFrame) -> list[dict]:
    """Converts a DataFrame to JSON-safe dicts (replaces Pandas NaN with None)."""
    if df.empty:
        return []
    return df.astype(object).where(pd.notnull(df), None).to_dict(orient="records")


@app.post("/api/extract")
def extract_pdf(
    file: UploadFile = File(...),
    model_name: str = Form("qwen2.5:3b"),
    review_threshold: float = Form(0.85),
    force_reparse: bool = Form(False),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    saved_pdf = UPLOAD_DIR / file.filename
    with saved_pdf.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Defined with `def` (not `async def`) so FastAPI runs this blocking
        # Docling + Ollama pipeline in a worker thread without freezing the UI.
        verified_df, review_df = process_pdf(
            pdf_path=saved_pdf,
            output_dir=OUTPUT_DIR,
            model_name=model_name,
            review_threshold=review_threshold,
            force_reparse=force_reparse,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    stem = saved_pdf.stem
    verified_xlsx = f"{stem}_verified.xlsx" if not verified_df.empty else None
    review_xlsx = f"{stem}_needs_review.xlsx" if not review_df.empty else None

    return {
        "filename": file.filename,
        "verified": df_to_clean_records(verified_df),
        "review": df_to_clean_records(review_df),
        "verified_xlsx": verified_xlsx,
        "review_xlsx": review_xlsx,
    }


@app.get("/api/download/{filename}")
def download_excel(filename: str):
    file_path = (OUTPUT_DIR / filename).resolve()
    if not file_path.is_file() or file_path.parent != OUTPUT_DIR:
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@app.get("/", response_class=HTMLResponse)
def index():
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Adaptive PDF Pipeline</title>
  <style>
    :root {
      --bg: #1e1e1e;
      --panel: #252526;
      --border: #3c3c3c;
      --text: #cccccc;
      --text-bright: #ffffff;
      --accent: #0e639c;
      --accent-hover: #1177bb;
      --green: #2ea043;
      --amber: #d29922;
      --red: #f85149;
    }
    * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
    body { margin: 0; padding: 20px; background: var(--bg); color: var(--text); font-size: 13px; }
    h2 { margin: 0 0 16px; color: var(--text-bright); font-size: 18px; }
    .controls {
      display: flex; flex-wrap: wrap; gap: 12px; align-items: end;
      background: var(--panel); padding: 14px; border: 1px solid var(--border); border-radius: 6px;
    }
    .field { display: flex; flex-direction: column; gap: 4px; }
    label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; color: #999; }
    input[type="text"], input[type="number"], input[type="file"] {
      background: var(--bg); border: 1px solid var(--border); color: var(--text-bright);
      padding: 6px 8px; border-radius: 4px; font-size: 13px;
    }
    .checkbox-field { flex-direction: row; align-items: center; padding-bottom: 6px; gap: 6px; }
    button {
      background: var(--accent); color: white; border: none; padding: 7px 14px;
      border-radius: 4px; cursor: pointer; font-weight: 600;
    }
    button:hover { background: var(--accent-hover); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    .status { margin: 14px 0; font-weight: 500; }
    .toolbar {
      display: none; justify-content: space-between; align-items: center;
      margin-top: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px;
    }
    .tabs { display: flex; gap: 8px; }
    .tab {
      background: var(--panel); color: var(--text); border: 1px solid var(--border);
      padding: 6px 12px; border-radius: 4px; cursor: pointer;
    }
    .tab.active { background: var(--accent); color: white; border-color: var(--accent); }
    .downloads a {
      color: #4ea1ff; text-decoration: none; margin-left: 12px; font-weight: 500;
    }
    .downloads a:hover { text-decoration: underline; }
    .table-wrap {
      margin-top: 12px; overflow-x: auto; border: 1px solid var(--border); border-radius: 6px;
      max-height: 70vh;
    }
    table { width: 100%; border-collapse: collapse; text-align: left; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); white-space: nowrap; }
    th { background: var(--panel); color: var(--text-bright); position: sticky; top: 0; z-index: 1; }
    tr:hover td { background: #2a2d2e; }
    td.wrap { white-space: normal; min-width: 200px; }
    .badge {
      padding: 2px 6px; border-radius: 4px; font-weight: 600; font-size: 11px; color: #fff;
    }
    .badge-high { background: var(--green); }
    .badge-mid { background: var(--amber); }
    .badge-low { background: var(--red); }
    .flag-text { color: #ff7b72; font-size: 12px; }
  </style>
</head>
<body>
  <h2>Adaptive PDF Structured Extraction</h2>

  <form id="uploadForm" class="controls">
    <div class="field">
      <label>PDF Document</label>
      <input type="file" id="pdfFile" accept=".pdf" required />
    </div>
    <div class="field">
      <label>Ollama Model</label>
      <input type="text" id="modelName" value="qwen2.5:3b" />
    </div>
    <div class="field">
      <label>Confidence Threshold</label>
      <input type="number" id="threshold" value="0.85" step="0.05" min="0" max="1" style="width: 90px;" />
    </div>
    <div class="field checkbox-field">
      <input type="checkbox" id="forceReparse" />
      <label for="forceReparse" style="text-transform:none; font-size:13px; color:var(--text);">Force OCR Reparse</label>
    </div>
    <button type="submit" id="runBtn">Extract Tables</button>
  </form>

  <div id="status" class="status"></div>

  <div id="toolbar" class="toolbar">
    <div class="tabs">
      <button type="button" class="tab active" id="tabVerified" onclick="switchTab('verified')">
        Auto-Verified (<span id="countVerified">0</span>)
      </button>
      <button type="button" class="tab" id="tabReview" onclick="switchTab('review')">
        Needs Review (<span id="countReview">0</span>)
      </button>
    </div>
    <div class="downloads" id="downloads"></div>
  </div>

  <div id="tableContainer" class="table-wrap" style="display:none;"></div>

  <script>
    let currentData = { verified: [], review: [] };
    let activeTab = 'verified';

    const COLUMNS = [
      { key: 'page_number', label: 'Page' },
      { key: 'item_code', label: 'Item Code' },
      { key: 'description', label: 'Description', wrap: true },
      { key: 'quantity', label: 'Qty' },
      { key: 'unit_price', label: 'Unit Price' },
      { key: 'total_amount', label: 'Total' },
      { key: 'category', label: 'Category' },
      { key: 'confidence_score', label: 'Confidence' },
      { key: 'review_flags', label: 'Review Flags', wrap: true },
      { key: 'document_entity', label: 'Entity' },
      { key: 'document_date', label: 'Date' }
    ];

    function escapeHtml(val) {
      if (val === null || val === undefined) return '';
      return String(val)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    function formatConfidence(score) {
      if (typeof score !== 'number') return '';
      const cls = score >= 0.85 ? 'badge-high' : score >= 0.70 ? 'badge-mid' : 'badge-low';
      return `<span class="badge ${cls}">${score.toFixed(3)}</span>`;
    }

    function renderTable() {
      const rows = currentData[activeTab] || [];
      const container = document.getElementById('tableContainer');
      container.style.display = 'block';

      if (!rows.length) {
        container.innerHTML = `<div style="padding:16px; color:#888;">No rows in this category.</div>`;
        return;
      }

      const thead = `<thead><tr>${COLUMNS.map(c => `<th>${c.label}</th>`).join('')}</tr></thead>`;
      const tbody = rows.map(row => {
        const cells = COLUMNS.map(col => {
          const val = row[col.key];
          if (col.key === 'confidence_score') return `<td>${formatConfidence(val)}</td>`;
          if (col.key === 'review_flags') return `<td class="wrap flag-text">${escapeHtml(val)}</td>`;
          return `<td class="${col.wrap ? 'wrap' : ''}">${escapeHtml(val)}</td>`;
        }).join('');
        return `<tr>${cells}</tr>`;
      }).join('');

      container.innerHTML = `<table>${thead}<tbody>${tbody}</tbody></table>`;
    }

    function switchTab(tab) {
      activeTab = tab;
      document.getElementById('tabVerified').classList.toggle('active', tab === 'verified');
      document.getElementById('tabReview').classList.toggle('active', tab === 'review');
      renderTable();
    }

    document.getElementById('uploadForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const fileInput = document.getElementById('pdfFile');
      if (!fileInput.files.length) return;

      const btn = document.getElementById('runBtn');
      const status = document.getElementById('status');
      btn.disabled = true;
      status.textContent = 'Running Docling layout parser and Ollama extraction (check terminal for page progress)...';

      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('model_name', document.getElementById('modelName').value);
      formData.append('review_threshold', document.getElementById('threshold').value);
      formData.append('force_reparse', document.getElementById('forceReparse').checked);

      try {
        const res = await fetch('/api/extract', { method: 'POST', body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Extraction failed');

        currentData = { verified: data.verified, review: data.review };
        document.getElementById('countVerified').textContent = data.verified.length;
        document.getElementById('countReview').textContent = data.review.length;

        const dl = document.getElementById('downloads');
        dl.innerHTML = '';
        if (data.verified_xlsx) {
          dl.innerHTML += `<a href="/api/download/${encodeURIComponent(data.verified_xlsx)}">Download Verified .xlsx</a>`;
        }
        if (data.review_xlsx) {
          dl.innerHTML += `<a href="/api/download/${encodeURIComponent(data.review_xlsx)}">Download Review .xlsx</a>`;
        }

        document.getElementById('toolbar').style.display = 'flex';
        status.textContent = `Completed: ${data.filename}`;
        switchTab(data.verified.length ? 'verified' : 'review');
      } catch (err) {
        status.textContent = `Error: ${err.message}`;
      } finally {
        btn.disabled = false;
      }
    });
  </script>
</body>
</html>"""