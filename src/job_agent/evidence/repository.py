from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from job_agent.evidence.contracts import (
    EvidenceBundle,
    EvidenceRunManifest,
    EvidenceRunRef,
    EvidenceV2Error,
    LegacyEvidenceView,
    SourceCatalog,
)
from job_agent.evidence.paths import ARTIFACT_FILENAMES, EvidenceArtifactPaths
from job_agent.evidence.sources import sha256_bytes


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class EvidenceArtifactRepository:
    def commit(self, session_dir: Path, bundle: EvidenceBundle) -> EvidenceRunRef:
        session_dir = Path(session_dir).resolve()
        paths = EvidenceArtifactPaths(session_dir, bundle.manifest.run_id)
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        if (paths.run_dir / "manifest.json").exists():
            raise EvidenceV2Error("immutable_run_exists")
        self._verify_source_catalog(session_dir, bundle.sources)
        payloads = {
            "sources.json": bundle.sources.model_dump(mode="json"),
            "requirements.json": bundle.requirements.model_dump(mode="json"),
            "mapping.json": bundle.mapping.model_dump(mode="json"),
            "assessment.json": bundle.assessment.model_dump(mode="json"),
        }
        hashes: dict[str, str] = {}
        for filename, payload in payloads.items():
            raw = _json_bytes(payload)
            _atomic_bytes(paths.run_dir / filename, raw)
            hashes[filename] = sha256_bytes(raw)
        manifest = bundle.manifest.model_copy(update={"artifact_hashes": hashes})
        manifest_raw = _json_bytes(manifest.model_dump(mode="json"))
        _atomic_bytes(paths.run_dir / "manifest.json", manifest_raw)
        manifest_hash = sha256_bytes(manifest_raw)
        pointer = {
            "schema_version": 2,
            "run_id": manifest.run_id,
            "path": PurePosixPath(paths.run_dir.relative_to(session_dir)).as_posix(),
            "manifest_hash": manifest_hash,
        }
        _atomic_bytes(paths.current, _json_bytes(pointer))
        return EvidenceRunRef(
            run_id=manifest.run_id,
            relative_path=pointer["path"],
            manifest_hash=manifest_hash,
        )

    def load_current(self, session_dir: Path) -> EvidenceBundle | None:
        session_dir = Path(session_dir).resolve()
        current = session_dir / "evidence_current.json"
        if not current.exists():
            return None
        try:
            pointer = json.loads(current.read_text(encoding="utf-8"))
            run_id = str(pointer["run_id"])
            bundle = self.load_run(session_dir, run_id)
            manifest_raw = (EvidenceArtifactPaths(session_dir, run_id).run_dir / "manifest.json").read_bytes()
            if sha256_bytes(manifest_raw) != pointer["manifest_hash"]:
                raise EvidenceV2Error("current_manifest_hash_mismatch")
            return bundle
        except (KeyError, json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            raise EvidenceV2Error("invalid_current_pointer") from exc

    def load_run(self, session_dir: Path, run_id: str) -> EvidenceBundle:
        session_dir = Path(session_dir).resolve()
        paths = EvidenceArtifactPaths(session_dir, run_id)
        try:
            manifest = EvidenceRunManifest.model_validate_json(
                (paths.run_dir / "manifest.json").read_text(encoding="utf-8")
            )
            loaded: dict[str, object] = {}
            for filename in ARTIFACT_FILENAMES:
                raw = (paths.run_dir / filename).read_bytes()
                if sha256_bytes(raw) != manifest.artifact_hashes.get(filename):
                    raise EvidenceV2Error("artifact_hash_mismatch")
                loaded[filename] = json.loads(raw.decode("utf-8"))
            bundle = EvidenceBundle.model_validate(
                {
                    "manifest": manifest,
                    "sources": loaded["sources.json"],
                    "requirements": loaded["requirements.json"],
                    "mapping": loaded["mapping.json"],
                    "assessment": loaded["assessment.json"],
                }
            )
            self._verify_source_catalog(session_dir, bundle.sources)
            return bundle
        except FileNotFoundError as exc:
            raise EvidenceV2Error("evidence_run_incomplete") from exc
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            raise EvidenceV2Error("invalid_evidence_run") from exc

    def load_legacy_view(self, session_dir: Path) -> LegacyEvidenceView | None:
        session_dir = Path(session_dir)
        jd_path = session_dir / "01_jd_structured.json"
        evidence_path = session_dir / "02_evidence_mapping.json"
        fit_path = session_dir / "03_fit_verdict.json"
        if not any(path.exists() for path in (jd_path, evidence_path, fit_path)):
            return None
        try:
            jd = json.loads(jd_path.read_text(encoding="utf-8")) if jd_path.exists() else None
            evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else []
            fit = json.loads(fit_path.read_text(encoding="utf-8")) if fit_path.exists() else None
            if not isinstance(evidence, list):
                raise EvidenceV2Error("invalid_legacy_evidence")
            return LegacyEvidenceView(structured_jd=jd, evidence=evidence, fit_verdict=fit)
        except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as exc:
            raise EvidenceV2Error("invalid_legacy_evidence") from exc

    @staticmethod
    def _verify_source_catalog(session_dir: Path, catalog: SourceCatalog) -> None:
        for document in catalog.documents:
            relative = PurePosixPath(document.snapshot_path)
            if relative.is_absolute() or ".." in relative.parts:
                raise EvidenceV2Error("forbidden_source")
            path = (session_dir / Path(*relative.parts)).resolve()
            if not path.is_relative_to(session_dir) or not path.is_file():
                raise EvidenceV2Error("source_snapshot_missing")
            if sha256_bytes(path.read_bytes()) != document.content_hash:
                raise EvidenceV2Error("stale_source_snapshot")
