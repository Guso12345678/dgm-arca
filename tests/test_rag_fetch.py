"""Corpus download script: validation, resilience and the licence README (no network)."""

import pytest

from rag.fetch_corpus import fetch_one, load_sources, validate_download, write_corpus_readme

PDF = b"%PDF-1.7\n" + b"x" * 2000


class Response:
    def __init__(self, content=PDF, status=200):
        self.content, self.status = content, status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


SOURCE = {"id": "s", "title": "Ley", "url": "https://x/y.pdf", "filename": "ley.pdf"}


def test_validate_download_rejects_non_documents():
    validate_download(PDF, "a.pdf")
    with pytest.raises(ValueError, match="not a document"):
        validate_download(b"tiny", "a.pdf")
    with pytest.raises(ValueError, match="not a PDF"):
        validate_download(b"<html>" + b"x" * 2000, "a.pdf")
    with pytest.raises(ValueError, match="HTML"):
        validate_download(b"x" * 2000, "a.html")


def test_fetch_downloads_then_skips_then_forces(tmp_path):
    assert fetch_one(SOURCE, tmp_path, get=lambda *a, **k: Response())["status"] == "downloaded"
    assert (tmp_path / "ley.pdf").read_bytes() == PDF

    def boom(*a, **k):
        raise AssertionError("no request expected")

    assert fetch_one(SOURCE, tmp_path, get=boom)["status"] == "present"
    assert fetch_one(SOURCE, tmp_path, force=True, get=lambda *a, **k: Response())["status"] == (
        "downloaded"
    )


def test_fetch_reports_failures_instead_of_raising(tmp_path):
    forbidden = fetch_one(SOURCE, tmp_path, get=lambda *a, **k: Response(status=403))
    assert forbidden["status"] == "failed" and "403" in forbidden["error"]
    bot_page = fetch_one(SOURCE, tmp_path, get=lambda *a, **k: Response(b"<html>" + b"x" * 2000))
    assert bot_page["status"] == "failed"
    assert not (tmp_path / "ley.pdf").exists()  # a bad answer is never written as the document


def test_manual_sources_are_flagged(tmp_path):
    manual = {
        "id": "m",
        "title": "EBA",
        "url": None,
        "manual": True,
        "filename": "eba.pdf",
        "license_note": "download it from the EBA site",
    }
    result = fetch_one(manual, tmp_path)
    assert result["status"] == "manual" and "EBA" in result["note"]
    (tmp_path / "eba.pdf").write_bytes(PDF)
    assert fetch_one(manual, tmp_path)["status"] == "present"


def test_corpus_readme_lists_present_files_with_origin_and_licence(tmp_path):
    (tmp_path / "ley.pdf").write_bytes(PDF)
    sources = [{**SOURCE, "license_note": "BOE legal notice"}, {**SOURCE, "filename": "absent.pdf"}]
    text = write_corpus_readme(sources, tmp_path).read_text(encoding="utf-8")
    assert "ley.pdf" in text and "https://x/y.pdf" in text and "BOE legal notice" in text
    assert "absent.pdf" not in text


def test_the_shipped_manifest_is_well_formed():
    sources = load_sources()
    assert len(sources) >= 3
    for source in sources:
        assert {"id", "title", "filename", "license_note"} <= set(source)
        assert source.get("url") or source.get("manual")
    assert len({s["id"] for s in sources}) == len(sources)
    assert len({s["filename"] for s in sources}) == len(sources)


def test_fetch_falls_back_to_the_next_url_and_reports_which_one_worked(tmp_path):
    source = {**SOURCE, "fallback_urls": ["https://x/bad.pdf", "https://x/good.pdf"]}
    answers = {
        "https://x/y.pdf": Response(b"<html>" + b"x" * 2000),  # bot page where a PDF was expected
        "https://x/bad.pdf": Response(status=404),
        "https://x/good.pdf": Response(),
    }
    tried = []

    def get(url, **kwargs):
        tried.append(url)
        return answers[url]

    result = fetch_one(source, tmp_path, get=get)
    assert result["status"] == "downloaded" and result["used_url"] == "https://x/good.pdf"
    assert tried == ["https://x/y.pdf", "https://x/bad.pdf", "https://x/good.pdf"]


def test_when_every_url_fails_the_error_lists_each_attempt(tmp_path):
    source = {**SOURCE, "fallback_urls": ["https://x/b.pdf"]}
    result = fetch_one(source, tmp_path, get=lambda *a, **k: Response(status=403))
    assert result["status"] == "failed"
    assert "https://x/y.pdf" in result["error"] and "https://x/b.pdf" in result["error"]


def test_percentile_thresholds_and_cut_shares_are_consistent():
    import numpy as np

    from rag.plot_distances import percentile_table, percentile_thresholds

    distances = np.linspace(0.0, 1.0, 101)
    values = percentile_thresholds(distances, [70, 80])
    assert values[70] == pytest.approx(0.70) and values[80] == pytest.approx(0.80)
    share = percentile_table(distances, [values[70]])[0]["share_of_cuts"]
    assert share == pytest.approx(0.30, abs=0.02)  # p70 cuts about 30 % of the boundaries
    assert percentile_thresholds(np.zeros(0), [70]) == {70: 0.0}
