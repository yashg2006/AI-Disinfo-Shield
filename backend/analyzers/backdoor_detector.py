"""
ModelSentinel — Backdoor Indicator Detector
Identifies neural-level indicators of backdoor insertion, including
neuron outliers, dead neuron clusters, and sparsity anomalies.
"""

import numpy as np
from typing import Dict, List

from backend.model_parser import ModelArtifact

# Thresholds
OUTLIER_SIGMA = 3.0           # Weights beyond 3σ from layer mean
OUTLIER_RATIO_THRESHOLD = 0.05  # >5% outlier neurons = suspicious
DEAD_NEURON_THRESHOLD = 1e-6    # Near-zero weight magnitude
SPARSITY_GRADIENT_THRESHOLD = 0.3  # Max expected sparsity change between adjacent layers


class BackdoorDetector:
    """Detect neural-level indicators of backdoor insertion."""

    def analyze(self, artifact: ModelArtifact) -> Dict:
        """
        Run backdoor indicator analysis on all layers.

        Returns:
            Dict with 'layer_analysis', 'alerts', and 'summary'.
        """
        layer_analysis = {}
        alerts = []
        flagged_layers = 0

        # Per-layer analysis
        for name, layer in artifact.layers.items():
            w = layer.weights.flatten()

            if w.size < 10:
                continue

            result = self._analyze_layer(name, w)
            layer_analysis[name] = result

            if result["flagged"]:
                flagged_layers += 1
                alerts.extend(result["alerts"])

        # Cross-layer sparsity gradient analysis
        sparsity_alerts = self._sparsity_gradient_analysis(layer_analysis)
        alerts.extend(sparsity_alerts)

        # Weight magnitude distribution across all layers
        magnitude_alerts = self._global_magnitude_analysis(artifact)
        alerts.extend(magnitude_alerts)

        total_layers = len(layer_analysis)
        score = min(100, int((flagged_layers / max(total_layers, 1)) * 100 * 2.0) + len(sparsity_alerts) * 10)

        return {
            "analyzer": "backdoor_detector",
            "layer_analysis": {
                name: {k: v for k, v in data.items() if k != "alerts"}
                for name, data in layer_analysis.items()
            },
            "alerts": alerts,
            "summary": {
                "total_layers_analyzed": total_layers,
                "flagged_layers": flagged_layers,
                "score": min(100, score),
            },
        }

    def _analyze_layer(self, name: str, weights: np.ndarray) -> Dict:
        """Analyze a single layer for backdoor indicators."""
        mean = float(np.mean(weights))
        std = float(np.std(weights))
        layer_alerts = []
        flagged = False

        # --- Neuron outlier detection ---
        if std > 1e-8:
            z_scores = np.abs((weights - mean) / std)
            outlier_mask = z_scores > OUTLIER_SIGMA
            outlier_count = int(np.sum(outlier_mask))
            outlier_ratio = outlier_count / weights.size
            max_z = float(np.max(z_scores))

            # Get the actual outlier values for reporting
            outlier_values = weights[outlier_mask]
            top_outliers = sorted(
                np.abs(outlier_values).tolist(), reverse=True
            )[:5]
        else:
            outlier_count = 0
            outlier_ratio = 0.0
            max_z = 0.0
            top_outliers = []

        if outlier_ratio > OUTLIER_RATIO_THRESHOLD:
            flagged = True
            layer_alerts.append({
                "severity": "critical" if outlier_ratio > 0.10 else "warning",
                "type": "neuron_outliers",
                "message": f"Layer '{name}' has {outlier_ratio:.1%} of weights as outliers "
                           f"(>{OUTLIER_SIGMA}σ from mean). {outlier_count} parameters have "
                           f"abnormally large magnitudes (max z-score: {max_z:.2f}). "
                           f"Backdoor neurons typically exhibit disproportionately large "
                           f"weights to activate strongly on trigger inputs.",
                "value": outlier_ratio,
                "threshold": OUTLIER_RATIO_THRESHOLD,
            })

        # --- Dead neuron analysis ---
        dead_mask = np.abs(weights) < DEAD_NEURON_THRESHOLD
        dead_count = int(np.sum(dead_mask))
        dead_ratio = dead_count / weights.size

        # --- Sparsity ---
        near_zero_mask = np.abs(weights) < 0.01
        sparsity = float(np.sum(near_zero_mask) / weights.size)

        # --- Check for bimodal distribution (backdoor often creates two clusters) ---
        bimodal_score = self._bimodality_check(weights)
        if bimodal_score > 0.7:
            flagged = True
            layer_alerts.append({
                "severity": "warning",
                "type": "bimodal_weights",
                "message": f"Layer '{name}' shows a bimodal weight distribution "
                           f"(bimodality score: {bimodal_score:.3f}). This pattern can "
                           f"indicate two separate functional pathways — one for normal "
                           f"operation and one for a backdoor trigger response.",
                "value": bimodal_score,
                "threshold": 0.7,
            })

        return {
            "outlier_count": outlier_count,
            "outlier_ratio": round(outlier_ratio, 4),
            "max_z_score": round(max_z, 4),
            "top_outlier_magnitudes": [round(v, 6) for v in top_outliers],
            "dead_neuron_ratio": round(dead_ratio, 4),
            "sparsity": round(sparsity, 4),
            "bimodality_score": round(bimodal_score, 4),
            "flagged": flagged,
            "alerts": layer_alerts,
        }

    def _bimodality_check(self, weights: np.ndarray) -> float:
        """
        Quick bimodality coefficient using Sarle's method.
        BC = (skewness² + 1) / (kurtosis + 3 * (n-1)²/((n-2)(n-3)))
        Values > 0.555 suggest bimodality.
        """
        from scipy.stats import skew, kurtosis

        n = len(weights)
        if n < 10:
            return 0.0

        s = skew(weights)
        k = kurtosis(weights)  # excess kurtosis

        # Sarle's bimodality coefficient
        numerator = s ** 2 + 1
        denominator = k + 3  # adjusted kurtosis

        if denominator <= 0:
            return 0.0

        bc = numerator / denominator
        return min(float(bc), 1.0)

    def _sparsity_gradient_analysis(self, layer_analysis: Dict) -> List[Dict]:
        """Check for abnormal sparsity changes between adjacent layers."""
        alerts = []
        layers = list(layer_analysis.items())

        for i in range(1, len(layers)):
            prev_name, prev_data = layers[i - 1]
            curr_name, curr_data = layers[i]

            sparsity_change = abs(curr_data["sparsity"] - prev_data["sparsity"])

            if sparsity_change > SPARSITY_GRADIENT_THRESHOLD:
                alerts.append({
                    "severity": "warning",
                    "type": "sparsity_gradient_anomaly",
                    "message": f"Abnormal sparsity change between '{prev_name}' "
                               f"({prev_data['sparsity']:.1%}) and '{curr_name}' "
                               f"({curr_data['sparsity']:.1%}). A sudden {sparsity_change:.1%} "
                               f"shift suggests selective weight manipulation in one layer.",
                    "value": sparsity_change,
                    "threshold": SPARSITY_GRADIENT_THRESHOLD,
                })

        return alerts

    def _global_magnitude_analysis(self, artifact: ModelArtifact) -> List[Dict]:
        """Check global weight magnitude distribution for anomalies."""
        alerts = []

        # Collect per-layer max magnitudes
        max_magnitudes = {}
        for name, layer in artifact.layers.items():
            w = layer.weights.flatten()
            if w.size < 10:
                continue
            max_magnitudes[name] = float(np.max(np.abs(w)))

        if len(max_magnitudes) < 3:
            return alerts

        values = list(max_magnitudes.values())
        global_mean = np.mean(values)
        global_std = np.std(values)

        if global_std < 1e-8:
            return alerts

        # Flag layers whose max magnitude is an outlier compared to other layers
        for name, max_mag in max_magnitudes.items():
            z = (max_mag - global_mean) / global_std
            if z > 3.0:
                alerts.append({
                    "severity": "warning",
                    "type": "magnitude_outlier_layer",
                    "message": f"Layer '{name}' has an unusually high maximum weight magnitude "
                               f"({max_mag:.4f}, z-score: {z:.2f} across all layers). "
                               f"This layer's scale is significantly different from the rest "
                               f"of the model, which may indicate injected large-magnitude "
                               f"backdoor weights.",
                    "value": max_mag,
                    "threshold": global_mean + 3 * global_std,
                })

        return alerts
