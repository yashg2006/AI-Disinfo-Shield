"""
ModelSentinel — Entropy Analyzer
Detects high-entropy or structured noise in model weights that may indicate
hidden steganographic data or encrypted payloads.
"""

import numpy as np
from scipy.stats import entropy as shannon_entropy
from typing import Dict

from backend.model_parser import ModelArtifact

# Thresholds
HISTOGRAM_BINS = 256
BYTE_ENTROPY_HIGH_THRESHOLD = 7.5  # Near-random data (max = 8.0 for bytes)
BYTE_ENTROPY_LOW_THRESHOLD = 3.0   # Unusually structured
WEIGHT_ENTROPY_HIGH_THRESHOLD = 7.0
WEIGHT_ENTROPY_BASELINE_MIN = 3.5
WEIGHT_ENTROPY_BASELINE_MAX = 6.5


class EntropyAnalyzer:
    """Analyze weight entropy to detect hidden data or anomalous randomness."""

    def analyze(self, artifact: ModelArtifact) -> Dict:
        """
        Run entropy analysis on all layers.

        Returns:
            Dict with 'layer_entropy', 'alerts', and 'summary'.
        """
        layer_entropy = {}
        alerts = []
        flagged_layers = 0

        for name, layer in artifact.layers.items():
            w = layer.weights.flatten()

            if w.size < 32:
                continue

            # --- Weight histogram entropy ---
            hist, _ = np.histogram(w, bins=HISTOGRAM_BINS, density=False)
            hist = hist.astype(np.float64)
            hist_sum = hist.sum()
            if hist_sum > 0:
                hist_prob = hist / hist_sum
                weight_entropy = float(shannon_entropy(hist_prob, base=2))
            else:
                weight_entropy = 0.0

            # --- Byte-level entropy ---
            # Cast float32 weights to raw bytes and compute byte distribution
            raw_bytes = w.astype(np.float32).tobytes()
            byte_array = np.frombuffer(raw_bytes, dtype=np.uint8)
            byte_counts = np.bincount(byte_array, minlength=256).astype(np.float64)
            byte_total = byte_counts.sum()
            if byte_total > 0:
                byte_prob = byte_counts / byte_total
                byte_ent = float(shannon_entropy(byte_prob, base=2))
            else:
                byte_ent = 0.0

            # --- Per-element entropy (variance of information content) ---
            # Unique value ratio: steganography may reduce or increase unique values
            unique_count = len(np.unique(w[:min(100000, w.size)]))
            sample_size = min(100000, w.size)
            unique_ratio = unique_count / sample_size

            layer_flagged = False
            layer_alerts = []

            # Check byte entropy (main steganography indicator)
            if byte_ent > BYTE_ENTROPY_HIGH_THRESHOLD:
                layer_flagged = True
                layer_alerts.append({
                    "severity": "critical",
                    "type": "high_byte_entropy",
                    "message": f"Layer '{name}' has extremely high byte-level entropy "
                               f"({byte_ent:.3f} / 8.0 bits). This approaches the entropy of "
                               f"random data and strongly suggests embedded encrypted or "
                               f"compressed steganographic payload in the weight bytes.",
                    "value": byte_ent,
                    "threshold": BYTE_ENTROPY_HIGH_THRESHOLD,
                })
            elif byte_ent < BYTE_ENTROPY_LOW_THRESHOLD and w.size > 100:
                layer_alerts.append({
                    "severity": "info",
                    "type": "low_byte_entropy",
                    "message": f"Layer '{name}' has unusually low byte entropy ({byte_ent:.3f}). "
                               f"This may indicate heavy quantization or very constrained weights.",
                    "value": byte_ent,
                    "threshold": BYTE_ENTROPY_LOW_THRESHOLD,
                })

            # Check weight distribution entropy
            if weight_entropy > WEIGHT_ENTROPY_HIGH_THRESHOLD:
                layer_flagged = True
                layer_alerts.append({
                    "severity": "warning",
                    "type": "high_weight_entropy",
                    "message": f"Layer '{name}' weight distribution entropy is unusually high "
                               f"({weight_entropy:.3f} bits). Trained weights typically show "
                               f"concentrated distributions with entropy {WEIGHT_ENTROPY_BASELINE_MIN}-"
                               f"{WEIGHT_ENTROPY_BASELINE_MAX} bits.",
                    "value": weight_entropy,
                    "threshold": WEIGHT_ENTROPY_HIGH_THRESHOLD,
                })

            if layer_flagged:
                flagged_layers += 1

            alerts.extend(layer_alerts)

            layer_entropy[name] = {
                "weight_entropy": round(weight_entropy, 4),
                "byte_entropy": round(byte_ent, 4),
                "unique_ratio": round(unique_ratio, 6),
                "flagged": layer_flagged,
            }

        total_layers = len(layer_entropy)
        score = min(100, int((flagged_layers / max(total_layers, 1)) * 100 * 2.0))

        return {
            "analyzer": "entropy_analyzer",
            "layer_entropy": layer_entropy,
            "alerts": alerts,
            "summary": {
                "total_layers_analyzed": total_layers,
                "flagged_layers": flagged_layers,
                "score": score,
            },
        }
