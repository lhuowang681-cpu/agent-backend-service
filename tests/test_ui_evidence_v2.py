from pathlib import Path
from unittest.mock import MagicMock

from job_agent.ui.views import evidence_v2_view
from tests.evidence_v2_fixtures import (
    commit_single_atom_evidence_v2,
    commit_zero_atom_evidence_v2,
)


def test_default_view_shows_labels_without_internal_numbers(tmp_path: Path, monkeypatch):
    commit_zero_atom_evidence_v2(tmp_path)
    fake_st = MagicMock()
    fit_column = MagicMock()
    confidence_column = MagicMock()
    fake_st.columns.return_value = (fit_column, confidence_column)
    monkeypatch.setattr(evidence_v2_view, "st", fake_st)

    evidence_v2_view.render_evidence_v2_view(tmp_path)

    fit_column.metric.assert_called_once_with("匹配判断", "信息不足")
    confidence_column.metric.assert_called_once_with("证据可信度", "低")
    rendered = " ".join(str(call) for call in fake_st.mock_calls)
    assert "verified_coverage" not in rendered
    assert "confidence=" not in rendered
    assert "%" not in rendered


def test_detail_uses_validated_jd_and_source_quotes(tmp_path: Path, monkeypatch):
    commit_single_atom_evidence_v2(tmp_path)
    fake_st = MagicMock()
    fake_st.columns.return_value = (MagicMock(), MagicMock())
    fake_st.expander.return_value.__enter__.return_value = None
    monkeypatch.setattr(evidence_v2_view, "st", fake_st)

    evidence_v2_view.render_evidence_v2_view(tmp_path)

    written = [call.args[0] for call in fake_st.write.call_args_list]
    assert "需要独立设计评测" in written
    assert "> 独立设计并运行过模型评测" in written


def test_legacy_session_is_labeled_read_only(tmp_path: Path, monkeypatch):
    (tmp_path / "01_jd_structured.json").write_text("{}", encoding="utf-8")
    fake_st = MagicMock()
    monkeypatch.setattr(evidence_v2_view, "st", fake_st)

    evidence_v2_view.render_evidence_v2_view(tmp_path)

    message = fake_st.info.call_args.args[0]
    assert "旧版证据" in message
    assert "不改写旧文件" in message
