"""
ModelSentinel — Layer Correlation Analyzer
Detects unusual inter-layer correlations and spectral anomalies that may
indicate coordinated payload embedding or covert data channels.
"""

import numpy as np
from typing import Dict, List, Tuple

from backend.model_parser import ModelArtifact

# Thresholds
CROSS_LAYER_CORR_THRESHOLD = 0.5     # Suspicious if non-adjacent layers correlate
SPECTRAL_PEAK_SIGMA = 3.0            # Frequency peaks above 3σ noise floor
CLUSTER_UNIFORMITY_THRESHOLD = 0.85  # Weight clustering ratio indicating quantized encoding
MAX_WEIGHTS_FOR_FFT = 100000


class LayerCorrelationAnalyzer:
    """Detect unusual cross-layer correlations and spectral patterns."""

    def analyze(self, artifact: ModelArtifact) -> Dict:
        """
        Run correlation and spectral analysis on model layers.

        Returns:
            Dict with 'correlations', 'spectral', 'alerts', and 'summary'.
        """
        alerts = []
        flagged = False

        # --- Cross-layer correlation analysis ---
        corr_results = self._cross_layer_correlation(artifact)
        for result in corr_results.get("suspicious_pairs", []):
            flagged = True
            alerts.append({
                "severity": "warning",
                "type": "cross_layer_correlation",
                "message": f"Layers '{result['layer_a']}' and '{result['layer_b']}' show "
                           f"unusually high correlation ({result['correlation']:.4f}). "
                           f"Non-adjacent layers should not exhibit strong weight correlations. "
                           f"This may indicate a coordinated payload embedded across layers.",
                "value": result["correlation"],
                "threshold": CROSS_LAYER_CORR_THRESHOLD,
            })

        # --- Spectral analysis per layer ---
        spectral_results = {}
        for name, layer in artifact.layers.items():
            w = layer.weights.flatten()
            if w.size < 64:
                continue

            spectral = self._spectral_analysis(w, name)
            spectral_results[name] = spectral

            if spectral.get("has_anomalous_peaks", False):
                flagged = True
                alerts.append({
                    "severity": "warning",
                    "type": "spectral_anomaly",
                    "message": f"Layer '{name}' contains anomalous frequency peaks in its "
                               f"weight spectrum ({spectral['peak_count']} peaks above "
                               f"{SPECTRAL_PEAK_SIGMA}σ). Structured periodic patterns in weights "
                               f"may indicate an embedded data channel or encoded signal.",
                    "value": spectral["peak_count"],
                    "threshold": 0,
                })

        # --- Weight magnitude clustering ---
        cluster_results = {}
        for name, layer in artifact.layers.items():
            w = layer.weights.flatten()
            if w.size < 100:
                continue

            cluster = self._weight_clustering_analysis(w)
            cluster_results[name] = cluster

            if cluster.get("suspicious_quantization", False):
                flagged = True
                alerts.append({
                    "severity": "warning",
                    "type": "suspicious_quantization",
                    "message": f"Layer '{name}' weights cluster into {cluster['n_clusters']} "
                               f"discrete levels with {cluster['cluster_ratio']:.1%} of values "
                               f"near cluster centers. This may indicate a quantized message "
                               f"encoding scheme.",
                    "value": cluster["cluster_ratio"],
                    "threshold": CLUSTER_UNIFORMITY_THRESHOLD,
                })

        score = min(100, len(alerts) * 20) if flagged else 0

        return {
            "analyzer": "layer_correlation",
            "correlations": corr_results,
            "spectral": spectral_results,
            "clustering": cluster_results,
            "alerts": alerts,
            "summary": {
                "suspicious_correlations": len(corr_results.get("suspicious_pairs", [])),
                "spectral_anomalies": sum(
                    1 for s in spectral_results.values() if s.get("has_anomalous_peaks")
                ),
                "flagged": flagged,
                "score": score,
            },
        }

    def _cross_layer_correlation(self, artifact: ModelArtifact) -> Dict:
        """Compute pairwise Pearson correlation between layer weight vectors."""
        layer_names = [n for n, l in artifact.layers.items() if l.param_count >= 100]
        n = len(layer_names)

        if n < 2:
            return {"correlation_matrix": [], "suspicious_pairs": []}

        # Flatten all layers to same size (truncate to minimum)
        vectors = {}
        for name in layer_names:
            w = artifact.layers[name].weights.flatten()
            vectors[name] = w[:min(len(w), MAX_WEIGHTS_FOR_FFT)]

        suspicious_pairs = []
        correlation_matrix = []

        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    row.append(1.0)
                    continue

                a = vectors[layer_names[i]]
                b = vectors[layer_names[j]]
                min_len = min(len(a), len(b))
                if min_len < 10:
                    row.append(0.0)
                    continue

                corr = float(np.corrcoef(a[:min_len], b[:min_len])[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                row.append(round(corr, 4))

                # Flag non-adjacent layers with high correlation
                if abs(i - j) > 1 and abs(corr) > CROSS_LAYER_CORR_THRESHOLD:
                    suspicious_pairs.append({
                        "layer_a": layer_names[i],
                        "layer_b": layer_names[j],
                        "correlation": corr,
                        "distance": abs(i - j),
                    })

            correlation_matrix.append(row)

        return {
            "layer_names": layer_names,
            "correlation_matrix": correlation_matrix,
            "suspicious_pairs": suspicious_pairs,
        }

    def _spectral_analysis(self, weights: np.ndarray, name: str) -> Dict:
        """FFT analysis to detect periodic structures hidden in weights."""
        w = weights[:min(len(weights), MAX_WEIGHTS_FOR_FFT)]

        # Compute power spectrum
        fft_vals = np.fft.rfft(w)
        power_spectrum = np.abs(fft_vals) ** 2

        # Ignore DC component
        power_no_dc = power_spectrum[1:]

        if len(power_no_dc) < 10:
            return {"has_anomalous_peaks": False, "peak_count": 0}

        # Detect peaks above noise floor
        mean_power = np.mean(power_no_dc)
        std_power = np.std(power_no_dc)

        if std_power < 1e-10:
            return {
                "has_anomalous_peaks": False,
                "peak_count": 0,
                "mean_power": float(mean_power),
            }

        threshold = mean_power + SPECTRAL_PEAK_SIGMA * std_power
        peaks = np.where(power_no_dc > threshold)[0]

        # Filter: only flag if there are significant, isolated peaks
        # (natural spectra may have a few scattered peaks, but structured
        # data creates distinct, strong peaks)
        significant_peaks = [
            {"frequency_bin": int(p), "power": float(power_no_dc[p])}
            for p in peaks[:10]
            if power_no_dc[p] > threshold * 1.5
        ]

        return {
            "has_anomalous_peaks": len(significant_peaks) >= 3,
            "peak_count": len(significant_peaks),
            "mean_power": float(mean_power),
            "std_power": float(std_power),
            "top_peaks": significant_peaks[:5],
        }

    def _weight_clustering_analysis(self, weights: np.ndarray) -> Dict:
        """Check if weights cluster into discrete levels (quantized encoding)."""
        w = weights[:min(len(weights), 50000)]

        # Simple histogram-based clustering detection
        # If weights concentrate heavily in a few narrow bins, it suggests quantization
        hist, bin_edges = np.histogram(w, bins=100)
        total = hist.sum()

        if total == 0:
            return {"suspicious_quantization": False, "n_clusters": 0, "cluster_ratio": 0.0}

        # Sort bins by count, check if top N bins contain most values
        sorted_counts = np.sort(hist)[::-1]
        top_n = 10
        top_ratio = float(sorted_counts[:top_n].sum() / total)

        # Count distinct "clusters" (bins with significant counts)
        significant_bins = np.sum(hist > total * 0.005)

        suspicious = (
            top_ratio > CLUSTER_UNIFORMITY_THRESHOLD
            and significant_bins < 20
            and significant_bins >= 3
        )

        return {
            "suspicious_quantization": suspicious,
            "n_clusters": int(significant_bins),
            "cluster_ratio": top_ratio,
        }
