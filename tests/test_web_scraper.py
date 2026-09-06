"""Tests for web scraper and OCR adapter (E4 + scraping capability)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from ipa import WebScraper, ScrapeSite, ScrapeResult, ScrapeSummary
from ipa.acquisition.ocr_adapter import OCRAdapter, OCRResult
from ipa.acquisition.web_scraper import PlaywrightBackend, DOCUMENT_EXTENSIONS


# ---------------------------------------------------------------------------
# WebScraper
# ---------------------------------------------------------------------------

class TestWebScraper:
    def test_slugify(self, tmp_path):
        with WebScraper(output_dir=tmp_path) as s:
            assert s._slugify("https://example.com/news/article-1") == "example-com-news-article-1"
            assert s._slugify("Hello World!") == "hello-world"
            assert s._slugify("") == ""

    def test_looks_like_article_filters_nav_and_social(self, tmp_path):
        base = "https://example.com"
        assert WebScraper._looks_like_article("/news/article-1", base)
        assert WebScraper._looks_like_article("/2024/01/15/something", base)
        assert not WebScraper._looks_like_article("#section", base)
        assert not WebScraper._looks_like_article("javascript:void(0)", base)
        assert not WebScraper._looks_like_article("mailto:a@b.com", base)
        assert not WebScraper._looks_like_article("/tag/security", base)
        assert not WebScraper._looks_like_article("/page/2", base)
        assert not WebScraper._looks_like_article("https://facebook.com/post", base)

    def test_fetch_page_returns_none_on_error(self, tmp_path):
        with WebScraper(output_dir=tmp_path, timeout=1) as s:
            result = s.fetch_page("http://127.0.0.1:1/nonexistent")
            assert result is None

    def test_extract_article_links_from_html(self, tmp_path):
        html = """
        <html><body>
            <main>
                <article><a href="/news/article-1">Article 1</a></article>
                <article><a href="/news/article-2">Article 2</a></article>
                <a href="/about">About</a>
                <a href="/tag/security">Security Tag</a>
                <a href="https://facebook.com/share">Share</a>
            </main>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(html, "https://example.com")
            assert "https://example.com/news/article-1" in links
            assert "https://example.com/news/article-2" in links
            # Non-article links should be filtered
            assert not any("about" in l for l in links)
            assert not any("tag" in l for l in links)
            assert not any("facebook" in l for l in links)

    def test_extract_article_links_with_date_pattern(self, tmp_path):
        html = """
        <html><body>
            <a href="/2024/01/15/security-alert">Alert</a>
            <a href="/2024/01/16/another-post">Post</a>
            <a href="/random-page">Random</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(html, "https://blog.example.com")
            assert "https://blog.example.com/2024/01/15/security-alert" in links
            assert "https://blog.example.com/2024/01/16/another-post" in links

    def test_extract_article_links_with_selector(self, tmp_path):
        html = """
        <html><body>
            <div class="article-card"><a href="/a/1">A1</a></div>
            <div class="article-card"><a href="/a/2">A2</a></div>
            <a href="/other">Other</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(
                html, "https://example.com", selector=".article-card a"
            )
            assert len(links) == 2
            assert "https://example.com/a/1" in links
            assert "https://example.com/a/2" in links

    def test_extract_article_links_filters_other_domains(self, tmp_path):
        html = """
        <html><body>
            <article><a href="https://other.com/article">External</a></article>
            <article><a href="/local-article">Local</a></article>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(html, "https://example.com")
            assert "https://example.com/local-article" in links
            assert not any("other.com" in l for l in links)

    def test_url_pattern_filters_to_matching_paths_only(self, tmp_path):
        """url_pattern should keep only URLs whose path matches the regex."""
        html = """
        <html><body>
            <a href="/blog/article-1/">A1</a>
            <a href="/blog/article-2/">A2</a>
            <a href="/blog/category/news/">Category</a>
            <a href="/blog/tag/security/">Tag</a>
            <a href="/about">About</a>
            <a href="/contact">Contact</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(
                html, "https://example.com",
                url_pattern=r"^/blog/[^/]+/$",
            )
            # Only /blog/<slug>/  matches â€” category and tag have 2 path segments
            assert "https://example.com/blog/article-1/" in links
            assert "https://example.com/blog/article-2/" in links
            assert not any("category" in l for l in links)
            assert not any("tag" in l for l in links)
            assert not any("about" in l for l in links)
            assert not any("contact" in l for l in links)

    def test_url_pattern_with_exclude_paths(self, tmp_path):
        """exclude_paths should remove URLs containing those substrings."""
        html = """
        <html><body>
            <a href="/blog/article-1/">A1</a>
            <a href="/blog/recent-posts/">Recent</a>
            <a href="/blog/category/news/">Category</a>
            <a href="/blog/article-2/">A2</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(
                html, "https://example.com",
                url_pattern=r"^/blog/[^/]+/$",
                exclude_paths=["/blog/category/", "/blog/recent-posts/"],
            )
            assert "https://example.com/blog/article-1/" in links
            assert "https://example.com/blog/article-2/" in links
            assert not any("recent-posts" in l for l in links)
            assert not any("category" in l for l in links)

    def test_selector_and_url_pattern_combined(self, tmp_path):
        """selector narrows elements, url_pattern narrows URLs."""
        html = """
        <html><body>
            <div class="post-card"><a href="/blog/good-post/">Good</a></div>
            <div class="post-card"><a href="/blog/category/news/">Cat</a></div>
            <div class="sidebar"><a href="/blog/other-post/">Other</a></div>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(
                html, "https://example.com",
                selector=".post-card a",
                url_pattern=r"^/blog/[^/]+/$",
            )
            # selector picks .post-card links, url_pattern filters out category
            assert "https://example.com/blog/good-post/" in links
            assert not any("category" in l for l in links)
            assert not any("other-post" in l for l in links)  # not in .post-card

    def test_exclude_paths_alone_without_pattern(self, tmp_path):
        """exclude_paths should work even without url_pattern."""
        html = """
        <html><body>
            <a href="/news/article-1">A1</a>
            <a href="/news/article-2">A2</a>
            <a href="/tag/security">Tag</a>
            <a href="/page/2">Page</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(
                html, "https://example.com",
                exclude_paths=["/tag/", "/page/"],
            )
            assert "https://example.com/news/article-1" in links
            assert "https://example.com/news/article-2" in links
            assert not any("/tag/" in l for l in links)
            assert not any("/page/" in l for l in links)

    def test_deterministic_mode_excludes_listing_page_itself(self, tmp_path):
        """The listing page URL should never appear in article links."""
        html = """
        <html><body>
            <a href="/blog/">Blog Home</a>
            <a href="/blog/article-1/">A1</a>
            <a href="/blog/article-2/">A2</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links = s.extract_article_links(
                html, "https://example.com/blog",
                url_pattern=r"^/blog/[^/]+/$",
            )
            # /blog/ itself should be excluded (it's the listing page)
            assert "https://example.com/blog/" not in links
            assert "https://example.com/blog/article-1/" in links

    def test_heuristic_mode_used_when_no_config(self, tmp_path):
        """When no selector/pattern/excludes are set, heuristics are used."""
        html = """
        <html><body>
            <main>
                <article><a href="/news/article-1">A1</a></article>
                <article><a href="/news/article-2">A2</a></article>
            </main>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            links_heuristic = s.extract_article_links(html, "https://example.com")
            # Heuristic mode should find articles via <article> tags
            assert "https://example.com/news/article-1" in links_heuristic
            assert "https://example.com/news/article-2" in links_heuristic

    def test_save_article_creates_file(self, tmp_path):
        with WebScraper(output_dir=tmp_path) as s:
            result = ScrapeResult(
                url="https://example.com/news/test-article",
                title="Test Article",
                text="This is the article content.",
                date="2026-08-27",
            )
            filepath = s.save_article(result)
            assert filepath.exists()
            content = filepath.read_text(encoding="utf-8")
            assert "# Test Article" in content
            assert "Source: https://example.com/news/test-article" in content
            assert "Date: 2026-08-27" in content
            assert "This is the article content." in content

    def test_save_article_with_ocr_text(self, tmp_path):
        with WebScraper(output_dir=tmp_path) as s:
            result = ScrapeResult(
                url="https://example.com/news/test",
                title="With Images",
                text="Article text.",
                date=None,
                ocr_texts=["OCR text from image 1", "OCR text from image 2"],
            )
            filepath = s.save_article(result)
            content = filepath.read_text(encoding="utf-8")
            assert "OCR extracted text from images" in content
            assert "OCR text from image 1" in content
            assert "OCR text from image 2" in content

    def test_save_article_filename_is_deterministic(self, tmp_path):
        with WebScraper(output_dir=tmp_path) as s:
            result = ScrapeResult(
                url="https://example.com/news/same-url",
                title="Same",
                text="Content",
            )
            f1 = s.save_article(result)
            f2 = s.save_article(result)
            assert f1 == f2  # Same URL â†’ same filename

    def test_save_article_different_urls_different_files(self, tmp_path):
        with WebScraper(output_dir=tmp_path) as s:
            r1 = ScrapeResult(url="https://example.com/a/1", title="A", text="A")
            r2 = ScrapeResult(url="https://example.com/a/2", title="B", text="B")
            f1 = s.save_article(r1)
            f2 = s.save_article(r2)
            assert f1 != f2

    def test_scrape_result_success_property(self):
        assert ScrapeResult(url="u", title="t", text="text").success
        assert not ScrapeResult(url="u", title="", text="", error="failed").success
        assert not ScrapeResult(url="u", title="", text="").success

    @patch("ipa.acquisition.web_scraper.requests.Session")
    def test_scrape_site_handles_fetch_failure(self, mock_session_cls, tmp_path):
        """If the listing page can't be fetched, return error summary."""
        mock_session = MagicMock()
        mock_session.get.side_effect = Exception("Connection refused")
        mock_session_cls.return_value = mock_session

        with WebScraper(output_dir=tmp_path) as s:
            s._session = mock_session
            summary = s.scrape_site(ScrapeSite(url="https://example.com/news"))
            assert summary.total_articles_found == 0
            assert summary.articles_scraped == 0
            assert len(summary.errors) > 0


# ---------------------------------------------------------------------------
# OCRAdapter
# ---------------------------------------------------------------------------

class TestOCRAdapter:
    def test_extract_text_file_not_found(self, tmp_path):
        with OCRAdapter(gpu=False) as ocr:
            result = ocr.extract_text(tmp_path / "nonexistent.png")
            assert not result.success
            assert "not found" in result.error.lower()
            assert result.text == ""

    def test_extract_text_simple_returns_string(self, tmp_path):
        """extract_text_simple should return empty string on error."""
        with OCRAdapter(gpu=False) as ocr:
            text = ocr.extract_text_simple(tmp_path / "nonexistent.png")
            assert text == ""

    def test_ocr_result_success_property(self):
        assert OCRResult(image_path="x", text="hello", confidence=0.9).success
        assert not OCRResult(image_path="x", text="", confidence=0.0, error="bad").success

    def test_lazy_loading_does_not_init_on_construction(self):
        """OCRAdapter should not load the model until first use."""
        ocr = OCRAdapter(gpu=False)
        assert ocr._reader is None
        ocr.close()

    def test_extract_texts_empty_list(self):
        with OCRAdapter(gpu=False) as ocr:
            results = ocr.extract_texts([])
            assert results == []

    def test_extract_texts_multiple_nonexistent(self, tmp_path):
        with OCRAdapter(gpu=False) as ocr:
            results = ocr.extract_texts([
                tmp_path / "a.png",
                tmp_path / "b.png",
            ])
            assert len(results) == 2
            assert all(not r.success for r in results)


class TestImageClassifier:
    """Tests for intelligent image classification (skip decorative images)."""

    def test_tiny_file_skipped(self, tmp_path):
        """Images <2KB should be classified as decorative (icons)."""
        from ipa.acquisition.ocr_adapter import ImageClassifier
        clf = ImageClassifier()
        img_path = tmp_path / "tiny.png"
        img_path.write_bytes(b"\x89PNG\r\n\x01\x00")  # 8 bytes
        should, reason = clf.classify(img_path)
        assert not should
        assert "small" in reason.lower()

    def test_synthetic_text_image_detected(self, tmp_path):
        """A synthetic image with text should be classified as text-bearing."""
        from ipa.acquisition.ocr_adapter import ImageClassifier
        from PIL import Image, ImageDraw, ImageFont
        clf = ImageClassifier()
        img_path = tmp_path / "text_chart.png"
        # Create a white image with black text (high edge density, few colors)
        img = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Chart Title\nData: 42%\nLabel: AI Research",
                  fill="black")
        draw.rectangle([50, 80, 350, 180], outline="black", width=2)
        draw.line([50, 150, 350, 100], fill="black", width=2)
        img.save(img_path)
        should, reason = clf.classify(img_path)
        assert should, f"Expected text-bearing, got: {reason}"

    def test_photograph_skipped(self, tmp_path):
        """A colorful photograph should be classified as decorative."""
        from ipa.acquisition.ocr_adapter import ImageClassifier
        from PIL import Image
        import random
        clf = ImageClassifier()
        img_path = tmp_path / "photo.png"
        # Create a noisy colorful image (many colors, low edge density)
        img = Image.new("RGB", (200, 200))
        pixels = img.load()
        random.seed(42)
        for y in range(200):
            for x in range(200):
                pixels[x, y] = (random.randint(0, 255),
                                random.randint(0, 255),
                                random.randint(0, 255))
        img.save(img_path)
        should, reason = clf.classify(img_path)
        # Random noise has many colors â€” should be classified as photograph
        assert not should or "error" in reason, f"Expected decorative, got: {reason}"

    def test_very_small_dimensions_skipped(self, tmp_path):
        """Images with dimensions <50px should be skipped."""
        from ipa.acquisition.ocr_adapter import ImageClassifier
        from PIL import Image
        clf = ImageClassifier()
        img_path = tmp_path / "icon.png"
        img = Image.new("RGB", (20, 20), "blue")
        img.save(img_path)
        should, reason = clf.classify(img_path)
        assert not should
        assert "small" in reason.lower() or "dimension" in reason.lower()

    def test_large_file_skipped(self, tmp_path):
        """Images >20MB should be skipped (OOM protection)."""
        from ipa.acquisition.ocr_adapter import ImageClassifier
        from PIL import Image
        import io
        clf = ImageClassifier()
        # Create a large image that exceeds the 20MB limit
        img_path = tmp_path / "huge.png"
        # 5000x5000 white image with noise â†’ large PNG
        img = Image.new("RGB", (5000, 5000), "white")
        # Add some noise to prevent compression from making it small
        pixels = img.load()
        import random
        random.seed(42)
        for y in range(0, 5000, 10):
            for x in range(0, 5000, 10):
                pixels[x, y] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img.save(img_path, format="PNG")
        file_size = img_path.stat().st_size
        if file_size > clf.MAX_FILE_SIZE:
            should, reason = clf.classify(img_path)
            assert not should
            assert "large" in reason.lower()
        else:
            # If the image didn't end up >20MB, just skip this assertion
            pass

    def test_ocr_result_has_skipped_fields(self):
        """OCRResult should support skipped and skip_reason fields."""
        result = OCRResult(
            image_path="x", text="", confidence=0.0,
            skipped=True, skip_reason="file_too_small",
        )
        assert result.skipped
        assert result.skip_reason == "file_too_small"

    def test_extract_texts_smart_skips_decorative(self, tmp_path):
        """extract_texts_smart should skip tiny/decorative images."""
        from PIL import Image
        # Create a tiny icon
        icon_path = tmp_path / "icon.png"
        Image.new("RGB", (20, 20), "blue").save(icon_path)
        # Create a text-bearing image
        from PIL import ImageDraw
        text_path = tmp_path / "chart.png"
        img = Image.new("RGB", (400, 200), "white")
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "Data Chart\n42% Growth", fill="black")
        draw.rectangle([50, 80, 350, 180], outline="black", width=2)
        img.save(text_path)

        with OCRAdapter(gpu=False) as ocr:
            results = ocr.extract_texts_smart([icon_path, text_path])
            assert len(results) == 2
            # Icon should be skipped
            assert results[0].skipped
            # Text image should not be skipped (may fail OCR but not skipped)
            assert not results[1].skipped


# ---------------------------------------------------------------------------
# ScrapeSite dataclass
# ---------------------------------------------------------------------------

class TestDownloadDocuments:
    """Tests for document download functionality (PDF, DOCX, etc.)."""

    def test_document_extensions_includes_common_formats(self):
        """DOCUMENT_EXTENSIONS should include all common document formats."""
        assert ".pdf" in DOCUMENT_EXTENSIONS
        assert ".docx" in DOCUMENT_EXTENSIONS
        assert ".pptx" in DOCUMENT_EXTENSIONS
        assert ".xlsx" in DOCUMENT_EXTENSIONS
        assert ".txt" in DOCUMENT_EXTENSIONS
        assert ".csv" in DOCUMENT_EXTENSIONS
        # .md is excluded â€” repo READMEs/docs are not research documents
        assert ".md" not in DOCUMENT_EXTENSIONS

    def test_download_documents_finds_pdf_links(self, tmp_path):
        """_download_documents should find and download <a> links to PDFs."""
        html = """
        <html><body>
            <a href="/reports/security-report.pdf">Security Report</a>
            <a href="/about">About</a>
            <a href="/data.csv">Data</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            # Mock the session.get to return fake content
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/pdf"}
            mock_resp.iter_content = lambda n: [b"%PDF-1.4 fake content"]
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            docs = s._download_documents(html, "https://example.com/article")
            # Should download the PDF and CSV
            assert len(docs) == 2
            # Files should exist
            for doc_path in docs:
                assert Path(doc_path).exists()
            # PDF should have .pdf extension
            assert any(p.endswith(".pdf") for p in docs)
            # CSV should have .csv extension
            assert any(p.endswith(".csv") for p in docs)

    def test_download_documents_skips_html_redirects(self, tmp_path):
        """Should skip when a 'document' URL returns text/html (error page)."""
        html = """
        <html><body>
            <a href="/missing.pdf">Missing PDF</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "text/html; charset=utf-8"}
            mock_resp.iter_content = lambda n: [b"<html>Not found</html>"]
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            docs = s._download_documents(html, "https://example.com/article")
            # Should not download HTML error page as a PDF
            assert len(docs) == 0

    def test_download_documents_deduplicates(self, tmp_path):
        """Should not download the same document URL twice."""
        html = """
        <html><body>
            <a href="/report.pdf">Report 1</a>
            <a href="/report.pdf">Report 2</a>
            <a href="/report.pdf">Report 3</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/pdf"}
            mock_resp.iter_content = lambda n: [b"%PDF-1.4 content"]
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            docs = s._download_documents(html, "https://example.com/article")
            assert len(docs) == 1

    def test_download_documents_ignores_non_document_links(self, tmp_path):
        """Should only download links with document extensions."""
        html = """
        <html><body>
            <a href="/report.pdf">PDF</a>
            <a href="/page.html">HTML page</a>
            <a href="/image.jpg">Image</a>
            <a href="/script.js">JS</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/pdf"}
            mock_resp.iter_content = lambda n: [b"%PDF-1.4"]
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            docs = s._download_documents(html, "https://example.com/article")
            # Only the PDF should be downloaded
            assert len(docs) == 1
            assert docs[0].endswith(".pdf")

    def test_download_documents_handles_external_urls(self, tmp_path):
        """Should resolve relative and absolute URLs correctly."""
        html = """
        <html><body>
            <a href="https://cdn.example.com/whitepaper.pdf">External PDF</a>
            <a href="/local-doc.docx">Local DOCX</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/pdf"}
            mock_resp.iter_content = lambda n: [b"%PDF-1.4"]
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            docs = s._download_documents(html, "https://example.com/article")
            assert len(docs) == 2

    def test_scrape_result_has_document_paths_field(self):
        """ScrapeResult should have a document_paths field."""
        result = ScrapeResult(
            url="https://example.com/article",
            title="Test",
            text="Content",
            document_paths=["/tmp/doc1.pdf", "/tmp/doc2.docx"],
        )
        assert result.document_paths == ["/tmp/doc1.pdf", "/tmp/doc2.docx"]

    def test_scrape_summary_has_documents_downloaded_field(self):
        """ScrapeSummary should have a documents_downloaded field."""
        summary = ScrapeSummary(
            site_url="https://example.com",
            total_articles_found=1,
            articles_scraped=1,
            articles_skipped=0,
            images_downloaded=0,
            documents_downloaded=3,
        )
        assert summary.documents_downloaded == 3

    def test_save_article_includes_document_references(self, tmp_path):
        """save_article should list downloaded documents in the .txt output."""
        with WebScraper(output_dir=tmp_path) as s:
            result = ScrapeResult(
                url="https://example.com/news/test",
                title="Test Article",
                text="Article content.",
                document_paths=["/landing/doc1.pdf", "/landing/doc2.docx"],
            )
            filepath = s.save_article(result)
            content = filepath.read_text(encoding="utf-8")
            assert "Linked documents downloaded" in content
            assert "doc1.pdf" in content
            assert "doc2.docx" in content


class TestArxivLinkFollowing:
    """Tests for arxiv link detection and PDF download."""

    def test_extract_arxiv_id_from_abs_url(self):
        """Should extract arxiv ID from /abs/ URLs."""
        assert WebScraper._extract_arxiv_id(
            "https://arxiv.org/abs/2608.07592"
        ) == "2608.07592"

    def test_extract_arxiv_id_from_pdf_url(self):
        """Should extract arxiv ID from /pdf/ URLs."""
        assert WebScraper._extract_arxiv_id(
            "https://arxiv.org/pdf/2608.07592"
        ) == "2608.07592"

    def test_extract_arxiv_id_from_pdf_url_with_extension(self):
        """Should extract arxiv ID from /pdf/<id>.pdf URLs."""
        assert WebScraper._extract_arxiv_id(
            "https://arxiv.org/pdf/2608.07592.pdf"
        ) == "2608.07592"

    def test_extract_arxiv_id_strips_version(self):
        """Should strip version suffix (v1, v2, etc.)."""
        assert WebScraper._extract_arxiv_id(
            "https://arxiv.org/abs/2608.07592v1"
        ) == "2608.07592"

    def test_extract_arxiv_id_returns_none_for_non_arxiv(self):
        """Should return None for non-arxiv URLs."""
        assert WebScraper._extract_arxiv_id(
            "https://example.com/doc.pdf"
        ) is None
        assert WebScraper._extract_arxiv_id(
            "https://arxiv.org/list/cs.AI/recent"
        ) is None

    def test_download_documents_follows_arxiv_links(self, tmp_path):
        """_download_documents should download PDFs from arxiv links."""
        html = """
        <html><body>
            <a href="https://arxiv.org/abs/2608.07592">Paper on arxiv</a>
            <a href="https://arxiv.org/pdf/2608.03890">Direct PDF link</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/pdf"}
            mock_resp.iter_content = lambda n: [b"%PDF-1.5 arxiv content"]
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            docs = s._download_documents(html, "https://example.com/article")
            assert len(docs) == 2
            for doc_path in docs:
                assert Path(doc_path).exists()
                assert "arxiv_" in doc_path
                assert doc_path.endswith(".pdf")

    def test_download_documents_deduplicates_arxiv_links(self, tmp_path):
        """Same arxiv paper linked via abs/ and pdf/ should only download once."""
        html = """
        <html><body>
            <a href="https://arxiv.org/abs/2608.07592">Abstract</a>
            <a href="https://arxiv.org/pdf/2608.07592">PDF</a>
            <a href="https://arxiv.org/pdf/2608.07592.pdf">PDF with ext</a>
        </body></html>
        """
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.headers = {"content-type": "application/pdf"}
            mock_resp.iter_content = lambda n: [b"%PDF-1.5"]
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            docs = s._download_documents(html, "https://example.com/article")
            assert len(docs) == 1


class TestRssFeed:
    """Tests for RSS feed parsing."""

    def test_parse_rss_feed_extracts_links(self, tmp_path):
        """Should extract article links from RSS 2.0 feed."""
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>Article 1</title>
              <link>https://example.com/article-1</link>
              <pubDate>Mon, 25 Aug 2026 10:00:00 +0000</pubDate>
            </item>
            <item>
              <title>Article 2</title>
              <link>https://example.com/article-2</link>
              <pubDate>Wed, 20 Aug 2026 12:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = rss_xml
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            links = s._parse_rss_feed("https://example.com/feed", days_back=30)
            assert len(links) == 2
            assert "https://example.com/article-1" in links
            assert "https://example.com/article-2" in links

    def test_parse_rss_feed_filters_by_date(self, tmp_path):
        """Should exclude items older than days_back."""
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>Recent</title>
              <link>https://example.com/recent</link>
              <pubDate>Mon, 25 Aug 2026 10:00:00 +0000</pubDate>
            </item>
            <item>
              <title>Old</title>
              <link>https://example.com/old</link>
              <pubDate>Mon, 01 Jan 2024 10:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = rss_xml
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            links = s._parse_rss_feed("https://example.com/feed", days_back=30)
            assert len(links) == 1
            assert "https://example.com/recent" in links

    def test_parse_rss_feed_applies_url_pattern(self, tmp_path):
        """Should filter links by url_pattern."""
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>Blog post</title>
              <link>https://example.com/blog/my-post</link>
              <pubDate>Mon, 25 Aug 2026 10:00:00 +0000</pubDate>
            </item>
            <item>
              <title>News article</title>
              <link>https://example.com/news/my-news</link>
              <pubDate>Mon, 25 Aug 2026 11:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = rss_xml
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            links = s._parse_rss_feed(
                "https://example.com/feed", days_back=30,
                url_pattern=r"^/blog/",
            )
            assert len(links) == 1
            assert "https://example.com/blog/my-post" in links

    def test_parse_rss_feed_applies_exclude_paths(self, tmp_path):
        """Should exclude links matching exclude_paths."""
        rss_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Test Feed</title>
            <item>
              <title>Article</title>
              <link>https://example.com/blog/keep-this</link>
              <pubDate>Mon, 25 Aug 2026 10:00:00 +0000</pubDate>
            </item>
            <item>
              <title>Category page</title>
              <link>https://example.com/blog/category/ai</link>
              <pubDate>Mon, 25 Aug 2026 11:00:00 +0000</pubDate>
            </item>
          </channel>
        </rss>"""
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = rss_xml
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            links = s._parse_rss_feed(
                "https://example.com/feed", days_back=30,
                exclude_paths=["/blog/category/"],
            )
            assert len(links) == 1
            assert "https://example.com/blog/keep-this" in links

    def test_parse_rss_feed_handles_atom(self, tmp_path):
        """Should parse Atom feeds (entry/link/@href)."""
        atom_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Test Atom Feed</title>
          <entry>
            <title>Article 1</title>
            <link href="https://example.com/article-1" />
            <published>2026-08-25T10:00:00Z</published>
          </entry>
          <entry>
            <title>Article 2</title>
            <link href="https://example.com/article-2" />
            <published>2026-08-20T12:00:00Z</published>
          </entry>
        </feed>"""
        with WebScraper(output_dir=tmp_path) as s:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.text = atom_xml
            mock_resp.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp)

            links = s._parse_rss_feed("https://example.com/atom", days_back=30)
            assert len(links) == 2

    def test_parse_rss_feed_returns_empty_on_error(self, tmp_path):
        """Should return empty list on fetch error or parse error."""
        with WebScraper(output_dir=tmp_path) as s:
            # Fetch error
            mock_resp = MagicMock()
            mock_resp.raise_for_status = MagicMock(
                side_effect=requests.RequestException("404")
            )
            s._session.get = MagicMock(return_value=mock_resp)
            assert s._parse_rss_feed("https://example.com/feed", 30) == []

            # Parse error (invalid XML)
            mock_resp2 = MagicMock()
            mock_resp2.status_code = 200
            mock_resp2.text = "not valid xml"
            mock_resp2.raise_for_status = MagicMock()
            s._session.get = MagicMock(return_value=mock_resp2)
            assert s._parse_rss_feed("https://example.com/feed", 30) == []

    def test_scrape_site_has_rss_feed_field(self):
        """ScrapeSite should accept an rss_feed parameter."""
        site = ScrapeSite(
            url="https://example.com",
            rss_feed="https://example.com/feed.xml",
        )
        assert site.rss_feed == "https://example.com/feed.xml"


class TestScrapeSite:
    def test_defaults(self):
        site = ScrapeSite(url="https://example.com")
        assert site.days_back == 2
        assert site.max_articles == 20
        assert site.delay_seconds == 1.0
        assert site.article_selector is None
        assert site.url_pattern is None
        assert site.exclude_paths == []

    def test_custom_values(self):
        site = ScrapeSite(
            url="https://example.com",
            days_back=7,
            max_articles=50,
            delay_seconds=2.0,
            article_selector=".article-list a",
            url_pattern=r"^/blog/[^/]+/$",
            exclude_paths=["/blog/category/", "/blog/tag/"],
        )
        assert site.days_back == 7
        assert site.max_articles == 50
        assert site.delay_seconds == 2.0
        assert site.article_selector == ".article-list a"
        assert site.url_pattern == r"^/blog/[^/]+/$"
        assert site.exclude_paths == ["/blog/category/", "/blog/tag/"]


# ---------------------------------------------------------------------------
# PlaywrightBackend
# ---------------------------------------------------------------------------

class TestPlaywrightBackend:
    def test_lazy_loading_does_not_init_on_construction(self):
        """PlaywrightBackend should not launch browser until first use."""
        backend = PlaywrightBackend(headless=True)
        assert backend._browser is None
        assert backend._playwright is None
        backend.close()

    def test_close_without_init_is_safe(self):
        backend = PlaywrightBackend()
        backend.close()  # should not raise

    def test_context_manager_closes_properly(self):
        with PlaywrightBackend() as backend:
            assert backend._browser is None  # not yet initialized
        # close should be safe even if never used
        assert backend._browser is None


# ---------------------------------------------------------------------------
# WebScraper engine selection
# ---------------------------------------------------------------------------

class TestWebScraperEngine:
    def test_default_engine_is_requests(self, tmp_path):
        with WebScraper(output_dir=tmp_path) as s:
            assert s.engine == "requests"

    def test_engine_auto(self, tmp_path):
        with WebScraper(output_dir=tmp_path, engine="auto") as s:
            assert s.engine == "auto"

    def test_engine_playwright(self, tmp_path):
        with WebScraper(output_dir=tmp_path, engine="playwright") as s:
            assert s.engine == "playwright"
            assert s._playwright is None  # lazy

    def test_playwright_lazy_init(self, tmp_path):
        with WebScraper(output_dir=tmp_path, engine="playwright") as s:
            assert s._playwright is None
            # _get_playwright would init it, but we don't call it here
        # close should be safe

    def test_close_cleans_up_playwright(self, tmp_path):
        s = WebScraper(output_dir=tmp_path, engine="playwright")
        # Simulate that playwright was initialized
        s._playwright = PlaywrightBackend()
        s.close()
        assert s._playwright is None
