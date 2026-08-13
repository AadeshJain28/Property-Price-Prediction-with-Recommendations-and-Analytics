"""Typed access to config/config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Repo root, resolved from this file rather than the working directory."""
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any] = field(repr=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        p = Path(path) if path else project_root() / "config" / "config.yaml"
        with open(p) as fh:
            raw = yaml.safe_load(fh)
        cfg = cls(raw=raw)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        w = self.raw["recommender"]["weights"]
        total = sum(w.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"recommender weights must sum to 1.0, got {total}")
        overlap = set(self.banned_features) & (
            set(self.numeric_features) | set(self.categorical_features)
        )
        if overlap:
            raise ValueError(f"banned features present in the feature list: {sorted(overlap)}")

    # -- convenience accessors -------------------------------------------------
    @property
    def target(self) -> str:
        return self.raw["target"]

    @property
    def banned_features(self) -> list[str]:
        return list(self.raw["banned_features"])

    @property
    def numeric_features(self) -> list[str]:
        return list(self.raw["numeric_features"])

    @property
    def categorical_features(self) -> list[str]:
        return list(self.raw["categorical_features"])

    @property
    def feature_names(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    def path(self, key: str) -> Path:
        return project_root() / self.raw["paths"][key]

    def data_path(self, key: str) -> Path:
        return project_root() / self.raw["data"][key]


def library_versions() -> dict[str, str]:
    """Versions that produced the artefact.

    A scikit-learn pickle is not portable across versions. Recording this makes a
    mismatch a legible message rather than an AttributeError from deep inside
    joblib -- which is how it presents when CI or a hosted app installs a newer
    scikit-learn than the one that did the training.
    """
    import platform

    versions = {"python": platform.python_version()}
    for name in ("sklearn", "numpy", "scipy", "pandas", "joblib", "xgboost"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            versions[name] = "absent"
    return versions
