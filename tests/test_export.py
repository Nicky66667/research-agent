import pytest
from tools.export import export_txt, export_docx, export_pdf

# mock report for testing (Markdown format, simulating real output)

SAMPLE_REPORT = """## Executive Summary
This paper examines transformer attention mechanisms and their impact on NLP tasks.

## Background
Transformers were introduced in 2017 by Vaswani et al. and have since become the dominant architecture.

## Key Findings
- Attention allows the model to focus on relevant tokens [Vaswani, 2017]
- Multi-head attention captures different types of relationships
- Self-attention scales quadratically with sequence length

## References
- Vaswani et al. (2017). Attention Is All You Need.
"""

SAMPLE_QUERY = "What are the key innovations in transformer attention mechanisms?"


# ── TXT ──────────────────────────────────────────────────────────

def test_txt_returns_bytes():
    result = export_txt(SAMPLE_REPORT, SAMPLE_QUERY)
    assert isinstance(result, bytes)

def test_txt_contains_query():
    result = export_txt(SAMPLE_REPORT, SAMPLE_QUERY).decode("utf-8")
    assert SAMPLE_QUERY in result

def test_txt_contains_content():
    result = export_txt(SAMPLE_REPORT, SAMPLE_QUERY).decode("utf-8")
    assert "transformer" in result.lower()
    assert "Vaswani" in result

def test_txt_has_header():
    result = export_txt(SAMPLE_REPORT, SAMPLE_QUERY).decode("utf-8")
    assert "Research Report" in result
    assert "Generated:" in result

def test_txt_empty_report():
    # Empty report should not raise an exception; return valid bytes instead
    result = export_txt("", "")
    assert isinstance(result, bytes)


# ── DOCX ─────────────────────────────────────────────────────────

def test_docx_returns_bytes():
    result = export_docx(SAMPLE_REPORT, SAMPLE_QUERY)
    assert isinstance(result, bytes)

def test_docx_is_valid_zip():
    # .docx is essentially a ZIP file; its magic bytes are PK\x03\x04
    result = export_docx(SAMPLE_REPORT, SAMPLE_QUERY)
    assert result[:4] == b"PK\x03\x04", "DOCX file should start with ZIP magic bytes"

def test_docx_not_empty():
    result = export_docx(SAMPLE_REPORT, SAMPLE_QUERY)
    assert len(result) > 1000  # 一个最小的 docx 至少几 KB

def test_docx_empty_report():
    result = export_docx("", "")
    assert isinstance(result, bytes)


# ── PDF ──────────────────────────────────────────────────────────

def test_pdf_returns_bytes():
    result = export_pdf(SAMPLE_REPORT, SAMPLE_QUERY)
    assert isinstance(result, bytes)

def test_pdf_has_magic_bytes():
    # PDF files always start with %PDF
    result = export_pdf(SAMPLE_REPORT, SAMPLE_QUERY)
    assert result[:4] == b"%PDF", "PDF file should start with %PDF"

def test_pdf_not_empty():
    result = export_pdf(SAMPLE_REPORT, SAMPLE_QUERY)
    assert len(result) > 1000

def test_pdf_empty_report():
    result = export_pdf("", "")
    assert isinstance(result, bytes)


# --- general edge cases ---

@pytest.mark.parametrize("export_fn", [export_txt, export_docx, export_pdf])
def test_no_query_arg(export_fn):
    # query parameter is optional; should not raise an error if missing
    result = export_fn(SAMPLE_REPORT)
    assert isinstance(result, bytes)

@pytest.mark.parametrize("export_fn", [export_txt, export_docx, export_pdf])
def test_special_characters(export_fn):
    # report should not crash when it contains special characters
    tricky = "## Title\nContent with <tags> & symbols: 100% valid\n- bullet"
    result = export_fn(tricky, "test query")
    assert isinstance(result, bytes)