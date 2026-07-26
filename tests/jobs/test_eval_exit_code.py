# tests/jobs/test_eval_exit_code.py
"""Тесты логики drift-детекции в eval.py."""

from unittest.mock import patch

import pytest


class TestDriftDetection:
    def test_exits_1_when_perplexity_above_threshold(self) -> None:
        """Perplexity выше порога означает деградацию, скрипт падает с exit(1)."""
        from scripts.eval import _check_drift

        with pytest.raises(SystemExit) as exc_info:
            _check_drift(
                metrics={"test_perplexity": 50.0, "test_loss": 3.9},
                drift_threshold=30.0,
                metric_key="test_perplexity",
            )
        assert exc_info.value.code == 1

    def test_does_not_exit_when_perplexity_below_threshold(self) -> None:
        """Perplexity ниже порога означает, что модель в норме."""
        from scripts.eval import _check_drift

        _check_drift(
            metrics={"test_perplexity": 15.0, "test_loss": 2.7},
            drift_threshold=30.0,
            metric_key="test_perplexity",
        )

    def test_exits_1_when_loss_above_threshold(self) -> None:
        """Проверка для test_loss (меньше — лучше)."""
        from scripts.eval import _check_drift

        with pytest.raises(SystemExit) as exc_info:
            _check_drift(
                metrics={"test_loss": 5.0},
                drift_threshold=3.0,
                metric_key="test_loss",
            )
        assert exc_info.value.code == 1

    def test_warns_when_metric_key_missing(self) -> None:
        """Если ключ метрики отсутствует, выводится warning без падения."""
        from scripts.eval import _check_drift

        with patch("logging.Logger.warning") as mock_warn:
            _check_drift(
                metrics={"test_loss": 2.0},
                drift_threshold=30.0,
                metric_key="test_perplexity",
            )
            mock_warn.assert_called()

    @pytest.mark.parametrize(
        "perplexity,threshold,should_exit",
        [
            (10.0, 30.0, False),
            (50.0, 30.0, True),
            (30.0, 30.0, False),
            (0.1, 30.0, False),
        ],
    )
    def test_various_perplexity_values(
        self, perplexity: float, threshold: float, should_exit: bool
    ) -> None:
        from scripts.eval import _check_drift

        if should_exit:
            with pytest.raises(SystemExit):
                _check_drift(
                    metrics={"test_perplexity": perplexity},
                    drift_threshold=threshold,
                    metric_key="test_perplexity",
                )
        else:
            _check_drift(
                metrics={"test_perplexity": perplexity},
                drift_threshold=threshold,
                metric_key="test_perplexity",
            )
