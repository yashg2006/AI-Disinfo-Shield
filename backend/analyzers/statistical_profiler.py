"""
ModelSentinel — Statistical Profiler Analyzer
Profiles per-layer weight distributions to detect statistical anomalies
that may indicate steganographic tampering or backdoor insertion.
"""

import numpy as np
from scipy import stats
from typing import Dict, List

from backend.model_parser import ModelArtifact

# Thresholds
SKEWNESS_THRESHOLD = 2.0
KURTOSIS_THRESHOLD = 7.0
NORMALITY_P_THRESHOLD = 0.01
SHAPIRO_SAMPLE_SIZE = 5000  # Shapiro-Wilk max sample size


class StatisticalProfiler:
    """Analyze per-layer weight distributions for anomalies."""

    def analyze(self, artifact: ModelArtifact) -> Dict:
        """
        Run statistical profiling on all layers.

        Returns:
            Dict with 'layer_stats', 'alerts', and 'summary'.
        """
        layer_stats = {}
        alerts = []
        flagged_layers = 0

        for name, layer in artifact.layers.items():
            w = layer.weights.flatten()

            if w.size < 10:
                # Skip very small tensors (e.g., single bias values)
                continue

            # Basic statistics
            mean = float(np.mean(w))
            std = float(np.std(w))
            minimum = float(np.min(w))
            maximum = float(np.max(w))
            median = float(np.median(w))

            # Distribution shape
            skewness = float(stats.skew(w))
            kurtosis = float(stats.kurtosis(w))  # Excess kurtosis

            # Normality test (subsample for performance)
            sample = w if w.size <= SHAPIRO_SAMPLE_SIZE else np.random.choice(w, SHAPIRO_SAMPLE_SIZE, replace=False)
            try:
                _, normality_p = stats.shapiro(sample)
                normality_p = float(normality_p)
            except Exception:
                normality_p = 1.0

            # Zero ratio (sparsity indicator)
            zero_ratio = float(np.sum(np.abs(w) < 1e-7) / w.size)

            layer_flagged = False
            layer_alerts = []

            # Check skewness
            if abs(skewness) > SKEWNESS_THRESHOLD:
                layer_flagged = True
                layer_alerts.append({
                    "severity": "warning",
                    "type": "high_skewness",
                    "message": f"Layer '{name}' has abnormal skewness ({skewness:.3f}). "
                               f"Trained weights are typically near-symmetric. "
                               f"This could indicate data injection or non-standard weight manipulation.",
                    "value": skewness,
                    "threshold": SKEWNESS_THRESHOLD,
                })

            # Check kurtosis
            if kurtosis > KURTOSIS_THRESHOLD:
                layer_flagged = True
                layer_alerts.append({
                    "severity": "warning",
                    "type": "high_kurtosis",
                    "message": f"Layer '{name}' has unusually high kurtosis ({kurtosis:.3f}). "
                               f"Heavy-tailed distributions may indicate injected outlier values "
                               f"used for backdoor pathways or encoded payloads.",
                    "value": kurtosis,
                    "threshold": KURTOSIS_THRESHOLD,
                })

            # Check normality failure
            if normality_p < NORMALITY_P_THRESHOLD and w.size >= 50:
                # Only flag if the deviation is strong
                layer_alerts.append({
                    "severity": "info",
                    "type": "non_normal_distribution",
                    "message": f"Layer '{name}' weight distribution significantly deviates "
                               f"from Gaussian (Shapiro-Wilk p={normality_p:.6f}). "
                               f"This is common but worth noting in context of other findings.",
                    "value": normality_p,
                    "threshold": NORMALITY_P_THRESHOLD,
                })

            if layer_flagged:
                flagged_layers += 1

            alerts.extend(layer_alerts)

            layer_stats[name] = {
                "param_count": layer.param_count,
                "shape": list(layer.shape),
                "mean": round(mean, 6),
                "std": round(std, 6),
                "min": round(minimum, 6),
                "max": round(maximum, 6),
                "median": round(median, 6),
                "skewness": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "normality_p": round(normality_p, 6),
                "zero_ratio": round(zero_ratio, 4),
                "flagged": layer_flagged,
            }

        total_layers = len(layer_stats)
        score = min(100, int((flagged_layers / max(total_layers, 1)) * 100 * 2.5))

        return {
            "analyzer": "statistical_profiler",
            "layer_stats": layer_stats,
            "alerts": alerts,
            "summary": {
                "total_layers_analyzed": total_layers,
                "flagged_layers": flagged_layers,
                "score": score,  # 0-100 contribution to risk
            },
        }
