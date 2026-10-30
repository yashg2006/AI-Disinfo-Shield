"""
ModelSentinel — LSB Steganography Detector
Detects Least Significant Bit steganographic encoding in model weight values
using chi-squared analysis, autocorrelation, and RS analysis techniques.
"""

import struct
import numpy as np
from scipy.stats import chisquare
from typing import Dict

from backend.model_parser import ModelArtifact

# Thresholds
CHI2_P_UNIFORM_THRESHOLD = 0.05  # p > 0.05 means data looks uniform (suspicious)
AUTOCORRELATION_THRESHOLD = 0.02  # Very low autocorrelation = random LSBs
RS_RATIO_THRESHOLD = 0.1  # Small difference between R and S groups = stego
MIN_WEIGHTS_FOR_LSB = 256  # Minimum weights needed for meaningful analysis


class LSBDetector:
    """Detect LSB-based steganographic encoding in model weights."""

    def analyze(self, artifact: ModelArtifact) -> Dict:
        """
        Run LSB steganography detection on all layers.

        Returns:
            Dict with 'layer_lsb', 'alerts', and 'summary'.
        """
        layer_lsb = {}
        alerts = []
        flagged_layers = 0

        for name, layer in artifact.layers.items():
            w = layer.weights.flatten()

            if w.size < MIN_WEIGHTS_FOR_LSB:
                continue

            # Extract LSBs from IEEE 754 float32 representation
            lsb_bits = self._extract_lsb_bits(w)

            # --- Chi-squared test for uniformity of LSBs ---
            chi2_result = self._chi_squared_test(lsb_bits)

            # --- Autocorrelation of LSB sequence ---
            autocorr = self._lsb_autocorrelation(lsb_bits)

            # --- RS Analysis ---
            rs_result = self._rs_analysis(w)

            # --- Bit-plane analysis: check multiple bit positions ---
            bit_plane_entropy = self._bit_plane_analysis(w)

            layer_flagged = False
            layer_alerts = []

            # Chi-squared: if p-value is HIGH, LSBs look uniform (= likely stego)
            if chi2_result["p_value"] > CHI2_P_UNIFORM_THRESHOLD and chi2_result["p_value"] > 0.3:
                layer_flagged = True
                severity = "critical" if chi2_result["p_value"] > 0.7 else "warning"
                layer_alerts.append({
                    "severity": severity,
                    "type": "lsb_uniform_distribution",
                    "message": f"Layer '{name}': LSBs follow a near-uniform distribution "
                               f"(chi² p-value={chi2_result['p_value']:.4f}). In natural trained "
                               f"weights, LSBs exhibit structure from the training process. "
                               f"Uniformly random LSBs strongly suggest steganographic data "
                               f"replacement in the least significant bits.",
                    "value": chi2_result["p_value"],
                    "threshold": CHI2_P_UNIFORM_THRESHOLD,
                })

            # Autocorrelation: very low = random sequence (stego)
            if abs(autocorr) < AUTOCORRELATION_THRESHOLD:
                # Only flag if chi-squared also suggests issues
                if chi2_result["p_value"] > CHI2_P_UNIFORM_THRESHOLD:
                    layer_flagged = True
                    layer_alerts.append({
                        "severity": "warning",
                        "type": "lsb_low_autocorrelation",
                        "message": f"Layer '{name}': LSB sequence has near-zero autocorrelation "
                                   f"({autocorr:.4f}), consistent with pseudo-random data. "
                                   f"Natural weights show mild LSB autocorrelation from gradient-based "
                                   f"optimization.",
                        "value": autocorr,
                        "threshold": AUTOCORRELATION_THRESHOLD,
                    })

            # RS Analysis: equal R and S groups = stego
            if rs_result["rs_difference"] < RS_RATIO_THRESHOLD:
                layer_flagged = True
                layer_alerts.append({
                    "severity": "warning",
                    "type": "rs_analysis_suspicious",
                    "message": f"Layer '{name}': RS analysis shows nearly equal Regular and "
                               f"Singular groups (diff={rs_result['rs_difference']:.4f}). "
                               f"This pattern is characteristic of LSB replacement steganography.",
                    "value": rs_result["rs_difference"],
                    "threshold": RS_RATIO_THRESHOLD,
                })

            if layer_flagged:
                flagged_layers += 1

            alerts.extend(layer_alerts)

            layer_lsb[name] = {
                "chi2_statistic": round(chi2_result["statistic"], 4),
                "chi2_p_value": round(chi2_result["p_value"], 6),
                "autocorrelation": round(autocorr, 6),
                "rs_regular_ratio": round(rs_result["regular_ratio"], 4),
                "rs_singular_ratio": round(rs_result["singular_ratio"], 4),
                "rs_difference": round(rs_result["rs_difference"], 4),
                "bit_plane_entropies": {
                    str(k): round(v, 4) for k, v in bit_plane_entropy.items()
                },
                "flagged": layer_flagged,
            }

        total_layers = len(layer_lsb)
        score = min(100, int((flagged_layers / max(total_layers, 1)) * 100 * 2.5))

        return {
            "analyzer": "lsb_detector",
            "layer_lsb": layer_lsb,
            "alerts": alerts,
            "summary": {
                "total_layers_analyzed": total_layers,
                "flagged_layers": flagged_layers,
                "score": score,
            },
        }

    def _extract_lsb_bits(self, weights: np.ndarray) -> np.ndarray:
        """Extract the least significant bit of each float32 weight's IEEE 754 representation."""
        # Convert float32 to uint32 bit pattern
        raw_bytes = weights.astype(np.float32).tobytes()
        uint32_view = np.frombuffer(raw_bytes, dtype=np.uint32)
        # Extract LSB
        lsb = uint32_view & 1
        return lsb.astype(np.int32)

    def _chi_squared_test(self, lsb_bits: np.ndarray) -> Dict:
        """
        Chi-squared goodness-of-fit test: are LSBs uniformly distributed?
        Under steganography, LSBs become uniformly random (50/50 split of 0s and 1s).
        """
        counts = np.bincount(lsb_bits, minlength=2).astype(np.float64)
        n = counts.sum()
        expected = np.array([n / 2, n / 2])

        try:
            statistic, p_value = chisquare(counts, f_exp=expected)
            return {"statistic": float(statistic), "p_value": float(p_value)}
        except Exception:
            return {"statistic": 0.0, "p_value": 0.0}

    def _lsb_autocorrelation(self, lsb_bits: np.ndarray, lag: int = 1) -> float:
        """
        Compute autocorrelation of LSB sequence at given lag.
        Natural weights show some autocorrelation; stego data does not.
        """
        n = len(lsb_bits)
        if n < lag + 10:
            return 1.0

        # Limit sample for performance
        sample = lsb_bits[:min(n, 500000)]
        mean = np.mean(sample.astype(np.float64))
        variance = np.var(sample.astype(np.float64))

        if variance < 1e-10:
            return 1.0

        shifted = sample[lag:]
        original = sample[:len(shifted)]
        autocorr = np.mean((original - mean) * (shifted - mean)) / variance
        return float(autocorr)

    def _rs_analysis(self, weights: np.ndarray, group_size: int = 4) -> Dict:
        """
        RS (Regular-Singular) analysis adapted from image steganalysis.
        Groups weights, applies a flipping function, and classifies groups
        as Regular (smoothness increases) or Singular (smoothness decreases).
        In clean models, R > S. In stego models, R ≈ S.
        """
        w = weights.flatten()[:min(len(weights.flatten()), 200000)]
        n_groups = len(w) // group_size

        if n_groups < 10:
            return {"regular_ratio": 0.5, "singular_ratio": 0.5, "rs_difference": 0.0}

        w_groups = w[:n_groups * group_size].reshape(n_groups, group_size)

        regular = 0
        singular = 0

        for group in w_groups:
            # Smoothness measure: sum of absolute differences
            smoothness_original = np.sum(np.abs(np.diff(group)))

            # Flip LSB of each weight's float representation
            flipped = self._flip_lsb_float(group)
            smoothness_flipped = np.sum(np.abs(np.diff(flipped)))

            if smoothness_flipped > smoothness_original:
                regular += 1
            elif smoothness_flipped < smoothness_original:
                singular += 1

        total = regular + singular
        if total == 0:
            return {"regular_ratio": 0.5, "singular_ratio": 0.5, "rs_difference": 0.0}

        r_ratio = regular / total
        s_ratio = singular / total

        return {
            "regular_ratio": r_ratio,
            "singular_ratio": s_ratio,
            "rs_difference": abs(r_ratio - s_ratio),
        }

    def _flip_lsb_float(self, values: np.ndarray) -> np.ndarray:
        """Flip the LSB of float32 values at the IEEE 754 bit level."""
        raw = values.astype(np.float32).tobytes()
        uint32 = np.frombuffer(raw, dtype=np.uint32).copy()
        uint32 ^= 1  # Flip LSB
        return np.frombuffer(uint32.tobytes(), dtype=np.float32)

    def _bit_plane_analysis(self, weights: np.ndarray, n_planes: int = 4) -> Dict[int, float]:
        """
        Compute entropy for multiple bit planes (bit 0 through bit n_planes-1).
        Stego typically affects the lowest bits most.
        """
        from scipy.stats import entropy as shannon_entropy

        w = weights.flatten()[:min(len(weights.flatten()), 200000)]
        raw = w.astype(np.float32).tobytes()
        uint32 = np.frombuffer(raw, dtype=np.uint32)

        results = {}
        for bit in range(n_planes):
            plane = (uint32 >> bit) & 1
            counts = np.bincount(plane, minlength=2).astype(np.float64)
            total = counts.sum()
            if total > 0:
                prob = counts / total
                ent = float(shannon_entropy(prob, base=2))
            else:
                ent = 0.0
            results[bit] = ent

        return results
