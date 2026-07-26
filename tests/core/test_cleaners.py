# tests/core/test_cleaners.py
"""
Тесты пайплайна очистки текста.
Чистая unit-логика без внешних зависимостей.
"""

import pytest

from src.core.data.cleaners import RegexCleaner, TextCleaningPipeline


class TestRegexCleaner:
    def test_removes_html_tags(self):
        cleaner = RegexCleaner(pattern="<.*?>", replacement=" ")
        assert cleaner.clean("<b>Hello</b> world") == " Hello  world"

    def test_removes_non_printable_characters(self):
        cleaner = RegexCleaner(pattern=r"[^\x20-\x7e\n\t]", replacement="")
        assert cleaner.clean("Hello\x00World\x1fClean") == "HelloWorldClean"

    def test_collapses_whitespace(self):
        cleaner = RegexCleaner(pattern=r"\s+", replacement=" ")
        assert cleaner.clean("too   many    spaces") == "too many spaces"

    def test_empty_string_returns_empty(self):
        cleaner = RegexCleaner(pattern="<.*?>", replacement=" ")
        assert cleaner.clean("") == ""

    def test_no_match_returns_original(self):
        cleaner = RegexCleaner(pattern="<.*?>", replacement=" ")
        original = "plain text without html"
        assert cleaner.clean(original) == original

    def test_replacement_is_applied(self):
        cleaner = RegexCleaner(pattern=r"\d+", replacement="NUM")
        assert cleaner.clean("order 123 placed") == "order NUM placed"

    def test_unicode_text_preserved(self):
        cleaner = RegexCleaner(pattern="<.*?>", replacement="")
        assert cleaner.clean("Привет <b>мир</b>") == "Привет мир"


class TestTextCleaningPipeline:
    def _make_pipeline(self) -> TextCleaningPipeline:
        return TextCleaningPipeline(
            cleaners=[
                RegexCleaner(pattern="<.*?>", replacement=" "),
                RegexCleaner(pattern="[^\\x20-\\x7e\\n\\t]", replacement=""),
                RegexCleaner(pattern=r"\s+", replacement=" "),
            ]
        )

    def test_pipeline_applies_all_steps_in_order(self):
        pipeline = self._make_pipeline()
        assert " Hello  world " not in pipeline("  <b>Hello</b>\\x00  world  ")

    def test_pipeline_with_empty_cleaners_is_identity(self):
        pipeline = TextCleaningPipeline(cleaners=[])
        assert pipeline("unchanged") == "unchanged"

    def test_pipeline_html_then_whitespace(self):
        pipeline = self._make_pipeline()
        result = pipeline("<p>  Breaking   news  </p>")
        assert "Breaking news" in result
        assert "<p>" not in result

    def test_pipeline_is_callable(self):
        assert callable(self._make_pipeline())

    def test_pipeline_single_cleaner(self):
        from src.core.data.cleaners import RegexCleaner, TextCleaningPipeline

        # Исправлено: r"\\d" заменено на r"\d"
        pipeline = TextCleaningPipeline(cleaners=[RegexCleaner(pattern=r"\d", replacement="")])
        assert pipeline("abc123def456") == "abcdef"

    @pytest.mark.parametrize(
        "text",
        [
            "FREE MONEY CLICK NOW!!!",
            "Normal message about project update.",
            "",
            "   ",
            "<html><body>Content</body></html>",
        ],
    )
    def test_pipeline_does_not_raise_on_various_inputs(self, text):
        pipeline = self._make_pipeline()
        assert isinstance(pipeline(text), str)
