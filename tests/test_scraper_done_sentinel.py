"""Tests for the scraper-done sentinel and orphan detection.

Regression coverage for the 2026-09-23 incident: the pipeline thread died
(dashboard restart) before writing Landing/web/.scraper_done, so the fast
path watch never got its idle gate and held the bulk embedding lease
forever (banner stuck at 100%, chat blocked, promotion gated).
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest

from ipa.acquisition import scrape_cli
from ipa.ingestion import fast_path_cli


class TestDoneFile:
    def _argv(self, *args: str) -> list[str]:
        return ["run_web_scrape.py", *args]

    def test_done_file_written_on_success(self, tmp_path, monkeypatch):
        done = tmp_path / ".scraper_done"
        out = tmp_path / "web"
        out.mkdir()
        summary = MagicMock(
            total_articles_found=1, articles_scraped=1,
            articles_skipped=0, errors=[], results=[],
        )
        scraper = MagicMock()
        scraper.scrape_site.return_value = summary
        fake_cls = MagicMock()
        fake_cls.return_value.__enter__.return_value = scraper
        monkeypatch.setattr(scrape_cli, "WebScraper", fake_cls)
        monkeypatch.setattr(sys, "argv", self._argv(
            "--url", "https://example.com/news", "--output", str(out),
            "--no-ocr", "--no-images", "--done-file", str(done),
        ))
        scrape_cli.main()
        assert done.read_text(encoding="utf-8") == "done"

    def test_done_file_written_on_failure(self, tmp_path, monkeypatch):
        """The sentinel fires even when the run fails — the gate means
        'nothing more is coming', not 'everything went well'."""
        done = tmp_path / ".scraper_done"
        monkeypatch.setattr(sys, "argv", self._argv(
            "--config", str(tmp_path / "missing.yaml"),
            "--done-file", str(done),
        ))
        with pytest.raises(SystemExit):
            scrape_cli.main()
        assert done.exists()

    def test_no_done_file_without_flag(self, tmp_path, monkeypatch):
        out = tmp_path / "web"
        out.mkdir()
        summary = MagicMock(
            total_articles_found=0, articles_scraped=0,
            articles_skipped=0, errors=[], results=[],
        )
        scraper = MagicMock()
        scraper.scrape_site.return_value = summary
        fake_cls = MagicMock()
        fake_cls.return_value.__enter__.return_value = scraper
        monkeypatch.setattr(scrape_cli, "WebScraper", fake_cls)
        monkeypatch.setattr(sys, "argv", self._argv(
            "--url", "https://example.com/news", "--output", str(out),
            "--no-ocr", "--no-images",
        ))
        scrape_cli.main()
        assert not (out / ".scraper_done").exists()


class TestParentAlive:
    def test_current_parent_is_alive(self):
        """pytest's parent process exists, so the check must say True."""
        assert fast_path_cli._parent_alive() is True


class TestFindRunningProcesses:
    def test_parses_cim_output(self, monkeypatch):
        """_find_running_processes parses `pid<TAB>cmdline` rows (the
        Get-CimInstance format that replaced the removed wmic)."""
        from ipa.dashboard import jobs

        result = MagicMock()
        result.stdout = (
            "124700\tpythonw.exe -u scripts/cli/run_fast_path.py --watch 10\n"
            "999999\tpythonw.exe -u scripts/cli/run_web_scrape.py\n"
        )
        fake = MagicMock(return_value=result)
        monkeypatch.setattr(jobs.subprocess, "run", fake)
        assert jobs._find_running_processes("run_fast_path.py") == [124700]
        assert jobs._find_running_processes("run_web_scrape.py") == [999999]
        assert jobs._find_running_processes("not_running.py") == []

    def test_returns_empty_on_command_failure(self, monkeypatch):
        from ipa.dashboard import jobs

        def boom(*a, **k):
            raise OSError("no shell")

        monkeypatch.setattr(jobs.subprocess, "run", boom)
        assert jobs._find_running_processes("run_fast_path.py") == []
