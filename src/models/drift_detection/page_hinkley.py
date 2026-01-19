# https://riverml.xyz/dev/api/drift/PageHinkley/

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from river.drift import PageHinkley


@dataclass
class PageHinkleyConfig:
    min_instances: int = 30
    delta: float = 0.005
    threshold: float = 50.0
    alpha: float = 1.0  # forgetting factor, closer to 1 => slower forgetting


class PageHinkleyDriftDetector:
    def __init__(self, config: PageHinkleyConfig = PageHinkleyConfig()):
        self.cfg = config
        self.detector = PageHinkley(
            min_instances=self.cfg.min_instances,
            delta=self.cfg.delta,
            threshold=self.cfg.threshold,
            alpha=self.cfg.alpha,
        )
        self.n_seen: int = 0
        self.last_drift_at: Optional[int] = None

    def update(self, value: float) -> bool:
        self.n_seen += 1
        in_drift = self.detector.update(value)
        if in_drift:
            self.last_drift_at = self.n_seen
        return bool(in_drift)

    def reset(self) -> None:
        cfg = self.cfg
        self.__init__(cfg)

    def info(self) -> Dict[str, Any]:
        return {
            "n_seen": self.n_seen,
            "last_drift_at": self.last_drift_at,
        }
