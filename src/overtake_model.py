# src/overtake_model.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import warnings
import pickle
import numpy as np

from sklearn.feature_extraction import DictVectorizer

warnings.filterwarnings("ignore", category=FutureWarning, module="pandas.*")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class OvertakeModel:
    """
    Calibrated pass-probability model.

    Features expected in `context` for prediction:
      - pace_gap (float, defender_prev - attacker_prev, seconds; + => attacker faster)
      - drs_available (bool)
      - straight_len (float; optional, proxy like speed_bias)
      - track_type (str; optional)
      - drs_bucket (str; one of {'drs_0','drs_1','drs_2plus','unknown'})
    Plus the attacker/defender driver IDs (strings).

    Deterministic mode: pass_decision = (logit >= 0)
    Stochastic mode:    pass_decision ~ Bernoulli(sigmoid(logit))
    """
    def __init__(self, vectorizer: DictVectorizer, coef: np.ndarray, intercept: float):
        self.vec = vectorizer
        self.coef = coef.reshape(1, -1)
        self.intercept = float(intercept)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        # safe sigmoid
        z = np.clip(x, -40, 40)
        return 1.0 / (1.0 + np.exp(-z))

    def _row_to_features(self, attacker: str, defender: str, ctx: Dict[str, Any]) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        d["pace_gap"] = float(ctx.get("pace_gap", 0.0))
        d["straight_len"] = float(ctx.get("straight_len", 0.0))
        d["drs_available"] = bool(ctx.get("drs_available", False))

        tt = str(ctx.get("track_type", "unknown"))
        db = str(ctx.get("drs_bucket", "unknown"))
        d[f"track_type={tt}"] = 1.0
        d[f"drs_bucket={db}"] = 1.0

        d[f"attacker={attacker}"] = 1.0
        d[f"defender={defender}"] = 1.0
        return d

    def predict_proba(self, attacker: str, defender: str, ctx: Dict[str, Any]) -> float:
        feats = self._row_to_features(attacker, defender, ctx)
        X = self.vec.transform([feats])  # sparse (1×n)
        logit_mat = X.dot(self.coef.T)  # (1×1) dense ndarray
        logit = float(np.asarray(logit_mat).ravel()[0]) + self.intercept
        return float(self._sigmoid(np.array([logit]))[0])

    def decide(self, attacker, defender, ctx, deterministic=True, rng=None) -> bool:
        p = self.predict_proba(attacker, defender, ctx)
        if deterministic:
            return p >= 0.5
        rng = rng or np.random.default_rng(0)
        return bool(rng.random() < p)


# ------------- Loading saved model (from traits.py) -------------
def load_calibrated_model(path: Optional[Path] = None) -> Optional[OvertakeModel]:
    """
    Load the persisted overtake model bundle saved by traits.estimate_driver_traits().
    Default path: outputs/traits/overtake_model.pkl
    """
    if path is None:
        path = _project_root() / "outputs" / "traits" / "overtake_model.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    vec = bundle["vectorizer"]
    coef = np.asarray(bundle["coef_"])
    intercept = float(bundle["intercept_"])
    return OvertakeModel(vec, coef, intercept)


# ------------- Example usage -------------
if __name__ == "__main__":
    model = load_calibrated_model()
    if model is None:
        print("[WARN] No calibrated overtake model found. Run src/traits.py first.")
    else:
        # Demo: build a context and make both decisions
        ctx = {
            "pace_gap": 0.35,          # attacker faster by 0.35s on prev lap
            "drs_available": True,
            "track_type": "permanent",
            "drs_bucket": "drs_2plus",
            "straight_len": 1.2        # arbitrary proxy (e.g., speed_bias)
        }
        pa = model.predict_proba("44", "16", ctx)  # example driver codes
        print(f"[INFO] P(pass) = {pa:.3f}")
        print(f"[INFO] Deterministic decision: {model.decide('44', '16', ctx, deterministic=True)}")
        print(f"[INFO] Stochastic sample:      {model.decide('44', '16', ctx, deterministic=False)}")
