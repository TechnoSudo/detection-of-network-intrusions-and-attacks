# https://riverml.xyz/dev/api/drift/ADWIN/

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from river.drift import ADWIN 


@dataclass
class ADWINConfig:
    delta: float = 0.002  # sensitivity (smaller => more sensitive)


class ADWINDriftDetector:
    def __init__(self, config: ADWINConfig = ADWINConfig()):
        self.cfg = config
        self.detector = ADWIN(delta=self.cfg.delta)
        self.n_seen: int = 0
        self.last_drift_at: Optional[int] = None

    def update(self, value: float) -> bool:
        """
        Returns True if drift detected at this update.
        """
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
            "width": getattr(self.detector, "width", None),
            "estimation": getattr(self.detector, "estimation", None),
        }