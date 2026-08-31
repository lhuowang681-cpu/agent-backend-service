from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Split = Literal["development", "held_out"]
Status = Literal["supported", "partial", "contradictory"]


@dataclass(frozen=True)
class AtomSpec:
    atom_id: str
    name: str
    quote: str
    importance: Literal["must", "should", "nice"] = "must"
    hard_gate: bool = False


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    split: Split
    slice_tags: tuple[str, ...]
    jd_text: str
    atoms: tuple[AtomSpec, ...]
    c1_artifact_type: str | None = None
    c2_artifact_type: str = "experiment_record"
    c3_artifact_type: str = "production_log"
    extra_source_noise: bool = False


def _atoms(*values: tuple) -> tuple[AtomSpec, ...]:
    return tuple(AtomSpec(*value) for value in values)


# All family IDs are new in v5. Families, not variants, are assigned to splits.
FAMILIES: tuple[FamilySpec, ...] = (
    FamilySpec(
        "benefits_only_sparse",
        "development",
        ("sparse_jd", "benefits_noise", "hardening_v5"),
        "我们提供弹性办公、导师制度和开放的技术氛围，欢迎投递。",
        (),
    ),
    FamilySpec(
        "observability_delivery_compound",
        "development",
        ("compound_parent", "atomization", "delivery"),
        "需要设计模型监控、配置灰度回滚并建立值班响应流程。",
        _atoms(
            ("model_monitoring", "模型监控", "设计模型监控"),
            ("canary_rollback", "灰度回滚", "配置灰度回滚"),
            ("oncall_response", "值班响应", "值班响应流程"),
        ),
    ),
    FamilySpec(
        "jd_repeated_anchor",
        "development",
        ("repeated_quote", "jd_anchor", "semantic_locator"),
        "Python 基础不作为单独要求；岗位要求 Python 工程化，并要求单元测试与类型检查。",
        _atoms(
            ("python_engineering", "Python 工程化", "Python 工程化"),
            ("unit_testing", "单元测试", "单元测试"),
            ("type_checking", "类型检查", "类型检查"),
        ),
    ),
    FamilySpec(
        "evidence_repeated_anchor",
        "development",
        ("repeated_quote", "evidence_anchor", "semantic_locator"),
        "要求构建检索评测集、分析召回失败并维护回归门槛。",
        _atoms(
            ("retrieval_set", "检索评测集", "构建检索评测集"),
            ("recall_failure", "召回失败分析", "分析召回失败"),
            ("regression_gate", "回归门槛", "维护回归门槛"),
        ),
        extra_source_noise=True,
    ),
    FamilySpec(
        "source_code_cap",
        "development",
        ("artifact_cap", "source_code", "c1_only"),
        "需要实现任务调度、处理幂等重试并编写并发测试。",
        _atoms(
            ("task_scheduler", "任务调度", "实现任务调度"),
            ("idempotent_retry", "幂等重试", "处理幂等重试"),
            ("concurrency_test", "并发测试", "并发测试"),
        ),
        c1_artifact_type="source_code",
    ),
    FamilySpec(
        "design_record_cap",
        "development",
        ("artifact_cap", "design_record", "c2"),
        "要求设计状态机、定义恢复语义并说明一致性边界。",
        _atoms(
            ("state_machine", "状态机设计", "设计状态机"),
            ("recovery_semantics", "恢复语义", "恢复语义"),
            ("consistency_boundary", "一致性边界", "一致性边界"),
        ),
        c2_artifact_type="design_record",
    ),
    FamilySpec(
        "metric_record_cap",
        "development",
        ("artifact_cap", "metric_record", "c3"),
        "需要降低接口延迟、控制错误率并验证容量水位。",
        _atoms(
            ("api_latency", "接口延迟", "降低接口延迟"),
            ("error_rate", "错误率", "控制错误率"),
            ("capacity", "容量水位", "容量水位"),
        ),
        c3_artifact_type="metric_record",
    ),
    FamilySpec(
        "contradictory_metric_record",
        "development",
        ("contradiction", "metric_record", "negative_evidence"),
        "要求上线缓存策略、稳定命中率并完成峰值压测。",
        _atoms(
            ("cache_rollout", "缓存策略上线", "上线缓存策略"),
            ("hit_rate", "命中率稳定", "稳定命中率"),
            ("peak_load", "峰值压测", "峰值压测"),
        ),
        c3_artifact_type="metric_record",
    ),
    FamilySpec(
        "mixed_artifact_caps",
        "development",
        ("artifact_cap", "multi_source", "mixed_levels"),
        "负责数据契约、离线实验和线上漂移监控。",
        _atoms(
            ("data_contract", "数据契约", "数据契约"),
            ("offline_experiment", "离线实验", "离线实验"),
            ("drift_monitoring", "漂移监控", "线上漂移监控"),
        ),
        c2_artifact_type="experiment_record",
        c3_artifact_type="production_log",
    ),
    FamilySpec(
        "unicode_nfc_locator",
        "development",
        ("unicode", "nfc", "semantic_locator"),
        "需要维护 Café 数据集、评测中文分词，并监控模型漂移 🧭。",
        _atoms(
            ("cafe_dataset", "Café 数据集", "Café 数据集"),
            ("zh_tokenization", "中文分词评测", "评测中文分词"),
            ("drift_compass", "模型漂移监控", "模型漂移 🧭"),
        ),
    ),
    FamilySpec(
        "punctuation_boundary",
        "development",
        ("exact_quote", "punctuation", "span_boundary"),
        "职责：（1）构造训练样本；（2）维护标签规范；（3）审计数据泄漏。",
        _atoms(
            ("training_samples", "训练样本", "构造训练样本"),
            ("label_contract", "标签规范", "维护标签规范"),
            ("leakage_audit", "数据泄漏审计", "审计数据泄漏"),
        ),
    ),
    FamilySpec(
        "multi_atom_hard_gate",
        "development",
        ("hard_gate", "multi_atom", "verified_gate"),
        "必须同时完成权限隔离和审计留痕；还需维护部署文档。",
        _atoms(
            ("permission_isolation", "权限隔离", "权限隔离", "must", True),
            ("audit_trail", "审计留痕", "审计留痕", "must", True),
            ("deploy_docs", "部署文档", "维护部署文档", "should", False),
        ),
    ),
    FamilySpec(
        "prefix_context_collision",
        "held_out",
        ("repeated_quote", "prefix_anchor", "heldout_v5"),
        "课程提到 Agent；岗位要求生产 Agent，并要求工具权限与失败恢复。",
        _atoms(
            ("production_agent", "生产 Agent", "生产 Agent"),
            ("tool_permission", "工具权限", "工具权限"),
            ("failure_recovery", "失败恢复", "失败恢复"),
        ),
    ),
    FamilySpec(
        "suffix_context_collision",
        "held_out",
        ("repeated_quote", "suffix_anchor", "heldout_v5"),
        "要求评测离线版本，而评测线上版本需要单独的指标告警与回滚策略。",
        _atoms(
            ("online_eval", "线上版本评测", "评测线上版本"),
            ("metric_alert", "指标告警", "指标告警"),
            ("rollback", "回滚策略", "回滚策略"),
        ),
    ),
    FamilySpec(
        "incident_report_cap",
        "held_out",
        ("artifact_cap", "incident_report", "c3"),
        "需要定位级联故障、执行止损并推动事故复盘闭环。",
        _atoms(
            ("cascade_failure", "级联故障定位", "定位级联故障"),
            ("mitigation", "故障止损", "执行止损"),
            ("incident_closure", "事故复盘闭环", "事故复盘闭环"),
        ),
        c3_artifact_type="incident_report",
    ),
    FamilySpec(
        "readme_cap",
        "held_out",
        ("artifact_cap", "readme", "c1_only"),
        "要求交付 SDK、维护兼容矩阵并提供最小示例。",
        _atoms(
            ("sdk_delivery", "SDK 交付", "交付 SDK"),
            ("compatibility", "兼容矩阵", "兼容矩阵"),
            ("minimal_example", "最小示例", "最小示例"),
        ),
        c1_artifact_type="readme",
    ),
    FamilySpec(
        "comparison_record_cap",
        "held_out",
        ("artifact_cap", "comparison_record", "c2"),
        "要求比较召回方案、验证重排收益并记录失败案例。",
        _atoms(
            ("retrieval_compare", "召回方案比较", "比较召回方案"),
            ("rerank_gain", "重排收益", "验证重排收益"),
            ("failure_cases", "失败案例", "失败案例"),
        ),
        c2_artifact_type="comparison_record",
    ),
    FamilySpec(
        "near_duplicate_claims",
        "held_out",
        ("near_duplicate", "deduplication", "multi_link"),
        "需要建立质量抽检、维护一致性指标并处理标注争议。",
        _atoms(
            ("quality_sampling", "质量抽检", "建立质量抽检"),
            ("agreement_metric", "一致性指标", "一致性指标"),
            ("label_dispute", "标注争议", "处理标注争议"),
        ),
        extra_source_noise=True,
    ),
    FamilySpec(
        "title_noise_sparse",
        "held_out",
        ("sparse_jd", "role_title_noise", "heldout_v5"),
        "岗位名称：大模型算法工程师。团队正在招聘，具体职责面谈。",
        (),
    ),
    FamilySpec(
        "mixed_language_delimiters",
        "held_out",
        ("bilingual", "delimiter", "unicode"),
        "Must own eval harness；负责 tool routing / retries；需要 production tracing。",
        _atoms(
            ("eval_harness", "eval harness", "own eval harness"),
            ("tool_retry", "tool routing retries", "tool routing / retries"),
            ("prod_tracing", "production tracing", "production tracing"),
        ),
    ),
)


PROFILES: tuple[tuple[tuple[Status, str] | None, ...], ...] = (
    (("supported", "C2"), ("supported", "C2"), ("supported", "C2")),
    (("supported", "C1"), ("supported", "C1"), ("supported", "C1")),
    (("partial", "C1"), ("supported", "C2"), None),
    (("contradictory", "C1"), ("supported", "C2"), ("supported", "C3")),
    (("supported", "C3"), None, None),
)


def _quote(atom: AtomSpec, status: Status, level: str, variant: int) -> str:
    if level == "C1":
        action = {
            "supported": f"本人独立设计并实现「{atom.name}」，测试与验收清单逐项通过并完成交付",
            "partial": f"仅协助「{atom.name}」的局部执行，未负责方案与验收",
            "contradictory": f"未参与也未负责「{atom.name}」",
        }[status]
        return action
    if level == "C2":
        action = {
            "supported": f"本人独立设计并实现「{atom.name}」，验收清单逐项通过；记录含 baseline、对照组、量化结果、失败样本与复现命令",
            "partial": f"针对「{atom.name}」只有单次试验，无对照组、复现命令与失败分析",
            "contradictory": f"针对「{atom.name}」的对照结果确认目标未完成",
        }[status]
        return action
    action = {
        "supported": f"「{atom.name}」已灰度运行 14 天，审计包含 p50/p95、错误率、告警、回滚演练和责任人",
        "partial": f"「{atom.name}」只完成小流量试运行，缺少完整指标与回滚验证",
        "contradictory": f"线上审计确认「{atom.name}」未部署且没有运行指标",
    }[status]
    return action


def _artifact_type(family: FamilySpec, level: str) -> str | None:
    if level == "C1":
        return family.c1_artifact_type
    if level == "C2":
        return family.c2_artifact_type
    return family.c3_artifact_type


def _fit_band(atoms: tuple[AtomSpec, ...], links: dict[str, list[dict]]) -> str:
    if len(atoms) < 3:
        return "insufficient_information"
    weights = {"must": 1.0, "should": 0.6, "nice": 0.2}
    values = {"supported": 1.0, "partial": 0.5, "contradictory": 0.0}
    numerator = sum(
        weights[atom.importance]
        * max((values[item["support_status"]] for item in links.get(atom.atom_id, [])), default=0.0)
        for atom in atoms
    )
    denominator = sum(weights[atom.importance] for atom in atoms)
    score = numerator / denominator
    return "high" if score >= 0.70 else "medium" if score >= 0.40 else "low"


def _case(family: FamilySpec, variant: int) -> dict:
    case_id = f"v5-{'dev' if family.split == 'development' else 'held'}-{family.family_id}-{variant + 1:02d}"
    if not family.atoms:
        return {
            "case_id": case_id,
            "family_id": family.family_id,
            "slice_tags": list(family.slice_tags),
            "jd_text": family.jd_text,
            "resume_text": f"候选材料 V{variant + 1} 仅包含课程兴趣与团队活动。",
            "evidence_sources": [],
            "gold_atoms": [],
            "gold_links": [],
            "expected_fit_band": "insufficient_information",
        }

    resume_parts: list[str] = []
    sources: list[dict] = []
    gold_links: list[dict] = []
    links_by_atom: dict[str, list[dict]] = {}
    for atom, recipe in zip(family.atoms, PROFILES[variant], strict=True):
        if recipe is None:
            continue
        status, level = recipe
        quote = _quote(atom, status, level, variant)
        artifact_type = _artifact_type(family, level)
        source_key = "resume"
        if level != "C1" or artifact_type is not None:
            source_key = f"{family.family_id}_{variant + 1}_{atom.atom_id}_{level.lower()}"
            noise = f"；术语索引：{atom.name} / {atom.name}" if family.extra_source_noise else ""
            source_prefix = {"C1": "原始材料", "C2": "实验记录", "C3": "运行审计"}[level]
            sources.append(
                {
                    "source_key": source_key,
                    "kind": "project_source" if artifact_type in {"source_code", "readme"} else "user_artifact",
                    "display_label": f"{artifact_type or 'unknown'} source",
                    "text": f"{source_prefix}V{variant + 1}：{quote}。材料由用户显式选择并保留上下文{noise}。",
                    **({"artifact_type": artifact_type} if artifact_type else {}),
                }
            )
        else:
            resume_parts.append(f"简历原文V{variant + 1}：{quote}")
        link = {
            "link_id": f"{atom.atom_id}-{status}-{level.lower()}-v{variant + 1}",
            "atom_id": atom.atom_id,
            "resume_quote": quote,
            "source_key": source_key,
            "support_status": status,
            "level": level,
        }
        gold_links.append(link)
        links_by_atom.setdefault(atom.atom_id, []).append(link)

    if not resume_parts:
        resume_parts.append(f"候选材料 V{variant + 1} 未额外声明岗位相关事实")
    return {
        "case_id": case_id,
        "family_id": family.family_id,
        "slice_tags": list(family.slice_tags),
        "jd_text": family.jd_text,
        "resume_text": "；".join(resume_parts) + "。",
        "evidence_sources": sources,
        "gold_atoms": [
            {
                "atom_id": atom.atom_id,
                "jd_quote": atom.quote,
                "importance": atom.importance,
                **({"hard_gate": True} if atom.hard_gate else {}),
            }
            for atom in family.atoms
        ],
        "gold_links": gold_links,
        "expected_fit_band": _fit_band(family.atoms, links_by_atom),
    }


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _variant(variant_id: str, corpus_id: str, corpus_hash: str) -> dict:
    versions = {
        "direct_llm_judge": ("direct-judge-v1", "direct-judge-v5", "direct-judge-v1"),
        "atomized_single_link": (
            "evidence-requirements-v2.1-quote-locator",
            "atomized-single-link-v5",
            "single-link-fit-v1",
        ),
        "full_evidence_v2": (
            "evidence-requirements-v2.1-quote-locator",
            "evidence-mapping-v2.2-support-level-separation",
            "fit_policy_v2_initial",
        ),
    }
    jd_version, prompt_version, fit_version = versions[variant_id]
    return {
        "variant_id": variant_id,
        "corpus_id": corpus_id,
        "corpus_hash": corpus_hash,
        "jd_parser_version": jd_version,
        "evidence_prompt_version": prompt_version,
        "fit_policy_version": fit_version,
        "provider": "anthropic_compatible",
        "model": "glm-5.2",
        "temperature": 0.0,
        "max_case_tokens": 4096,
    }


def build_artifacts() -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    audit: dict[str, object] = {
        "schema_version": 1,
        "corpus_revision": "evidence-v2-hardening-100-cases-v5",
        "created_at": "2026-08-01T20:00:00+08:00",
        "construction": "20 new scenario families x 5 evidence variants; family-disjoint 60/40 split",
        "synthetic_only": True,
        "v4_held_out_used_for_tuning": False,
        "quality_claim_allowed_only_after_frozen_v5_held_out": True,
        "splits": {},
    }
    for split, short in (("development", "dev"), ("held_out", "held_out")):
        families = [item for item in FAMILIES if item.split == split]
        cases = [_case(family, variant) for family in families for variant in range(5)]
        corpus_id = f"evidence-v2-hardening-{split}-corpus-v5"
        corpus = {"schema_version": 1, "corpus_id": corpus_id, "cases": cases}
        corpus_bytes = _json_bytes(corpus)
        corpus_hash = hashlib.sha256(corpus_bytes).hexdigest()
        corpus_name = f"cases_{short}_v5.json"
        manifest_name = f"manifest_{short}_v5.json"
        manifest = {
            "schema_version": 1,
            "manifest_id": f"evidence_v2_hardening_{short}_v5",
            "split": split,
            "corpus_id": corpus_id,
            "corpus_hash": corpus_hash,
            "metric_version": "evidence_metrics_v1",
            "case_inventory": [
                {
                    "case_id": case["case_id"],
                    "family_id": case["family_id"],
                    "slice_tags": case["slice_tags"],
                }
                for case in cases
            ],
            "required_slices": sorted({tag for case in cases for tag in case["slice_tags"]}),
            "variants": [
                _variant(variant_id, corpus_id, corpus_hash)
                for variant_id in ("direct_llm_judge", "atomized_single_link", "full_evidence_v2")
            ],
            "held_out_frozen": split == "held_out",
            "synthetic_only": True,
        }
        manifest_bytes = _json_bytes(manifest)
        artifacts[corpus_name] = corpus_bytes
        artifacts[manifest_name] = manifest_bytes
        audit["splits"][split] = {
            "case_count": len(cases),
            "family_count": len(families),
            "family_ids": [item.family_id for item in families],
            "corpus_file": corpus_name,
            "corpus_sha256": corpus_hash,
            "manifest_file": manifest_name,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "frozen_before_live_run": True,
        }
    artifacts["corpus_100_audit_v5.json"] = _json_bytes(audit)
    return artifacts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the frozen Evidence hardening v5 corpus")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "eval" / "evidence_v2",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    artifacts = build_artifacts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mismatches: list[str] = []
    for name, content in artifacts.items():
        path = args.output_dir / name
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                mismatches.append(name)
        else:
            path.write_bytes(content)
    if mismatches:
        raise SystemExit("generated v5 corpus differs: " + ", ".join(mismatches))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
