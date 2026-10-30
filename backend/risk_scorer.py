"""
ModelSentinel — Risk Scoring Engine
Aggregates findings from all 6 analyzers into a composite Model Risk Score (0-100)
with human-readable explanations and recommendations.
"""

from typing import Dict, List
from datetime import datetime, timezone


# Category weights for composite score
CATEGORY_WEIGHTS = {
    "statistical_profiler": 0.15,
    "entropy_analyzer": 0.20,
    "lsb_detector": 0.25,
    "layer_correlation": 0.10,
    "backdoor_detector": 0.15,
    "metadata_inspector": 0.15,
}

# Risk level thresholds
RISK_LEVELS = [
    (25, "low", "Low Risk", "✅", "Model appears clean with no significant anomalies detected."),
    (50, "moderate", "Moderate Risk", "⚠️", "Some anomalies detected that warrant review. Manual inspection recommended."),
    (75, "high", "High Risk", "🔶", "Multiple suspicious indicators found. Model should not be deployed without thorough audit."),
    (100, "critical", "Critical Risk", "🔴", "Strong indicators of tampering or malicious content. Do NOT deploy this model."),
]


class RiskScorer:
    """Aggregate analysis results into a composite risk score with explanations."""

    def compute(self, analysis_results: Dict[str, Dict], model_info: Dict) -> Dict:
        """
        Compute composite risk score from all analyzer outputs.

        Args:
            analysis_results: Dict mapping analyzer name to its result dict.
            model_info: Basic model metadata (name, format, params, etc.)

        Returns:
            Complete risk assessment report.
        """
        # Extract per-analyzer scores
        category_scores = {}
        for analyzer_name, weight in CATEGORY_WEIGHTS.items():
            result = analysis_results.get(analyzer_name, {})
            summary = result.get("summary", {})
            raw_score = summary.get("score", 0)
            category_scores[analyzer_name] = {
                "raw_score": raw_score,
                "weight": weight,
                "weighted_score": raw_score * weight,
            }

        # Composite score (weighted sum)
        composite_score = sum(
            cs["weighted_score"] for cs in category_scores.values()
        )
        composite_score = min(100, max(0, int(round(composite_score))))

        # Determine risk level
        risk_level = self._get_risk_level(composite_score)

        # Collect all alerts, sorted by severity
        all_alerts = []
        for result in analysis_results.values():
            all_alerts.extend(result.get("alerts", []))

        severity_order = {"critical": 0, "warning": 1, "info": 2}
        all_alerts.sort(key=lambda a: severity_order.get(a.get("severity", "info"), 3))

        # Count alerts by severity
        alert_counts = {
            "critical": sum(1 for a in all_alerts if a["severity"] == "critical"),
            "warning": sum(1 for a in all_alerts if a["severity"] == "warning"),
            "info": sum(1 for a in all_alerts if a["severity"] == "info"),
            "total": len(all_alerts),
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            composite_score, category_scores, all_alerts, model_info
        )

        # Build category breakdown for the frontend
        category_breakdown = []
        category_labels = {
            "statistical_profiler": {"name": "Statistical Profile", "icon": "📊"},
            "entropy_analyzer": {"name": "Entropy Analysis", "icon": "🎲"},
            "lsb_detector": {"name": "LSB Steganography", "icon": "🔍"},
            "layer_correlation": {"name": "Layer Correlations", "icon": "🔗"},
            "backdoor_detector": {"name": "Backdoor Indicators", "icon": "🚪"},
            "metadata_inspector": {"name": "Format & Metadata", "icon": "📋"},
        }

        for analyzer_name, scores in category_scores.items():
            label = category_labels.get(analyzer_name, {"name": analyzer_name, "icon": "❓"})
            category_breakdown.append({
                "id": analyzer_name,
                "name": label["name"],
                "icon": label["icon"],
                "score": scores["raw_score"],
                "weight": scores["weight"],
                "weighted_score": round(scores["weighted_score"], 1),
                "level": self._get_risk_level(scores["raw_score"])["level"],
            })

        return {
            "scan_timestamp": datetime.now(timezone.utc).isoformat(),
            "model_info": model_info,
            "risk_score": composite_score,
            "risk_level": risk_level,
            "category_scores": category_breakdown,
            "alert_counts": alert_counts,
            "alerts": all_alerts,
            "recommendations": recommendations,
            "detailed_results": {
                name: {k: v for k, v in result.items() if k != "alerts"}
                for name, result in analysis_results.items()
            },
        }

    def _get_risk_level(self, score: int) -> Dict:
        """Map a score to its risk level."""
        for threshold, level, label, icon, desc in RISK_LEVELS:
            if score <= threshold:
                return {
                    "level": level,
                    "label": label,
                    "icon": icon,
                    "description": desc,
                    "score": score,
                }
        # Fallback
        return {
            "level": "critical",
            "label": "Critical Risk",
            "icon": "🔴",
            "description": "Strong indicators of tampering detected.",
            "score": score,
        }

    def _generate_recommendations(
        self,
        score: int,
        category_scores: Dict,
        alerts: List[Dict],
        model_info: Dict,
    ) -> List[Dict]:
        """Generate actionable security recommendations based on findings."""
        recommendations = []

        # Format-based recommendation
        if model_info.get("model_format") == "pytorch":
            recommendations.append({
                "priority": "high",
                "category": "format_safety",
                "title": "Convert to SafeTensors Format",
                "description": "This model uses pickle-based PyTorch format which can "
                               "execute arbitrary code on load. Convert to SafeTensors "
                               "(.safetensors) to eliminate code execution risk while "
                               "preserving all model weights.",
                "action": "from safetensors.torch import save_file; "
                          "save_file(model.state_dict(), 'model.safetensors')",
            })

        # Score-based recommendations
        if score > 50:
            recommendations.append({
                "priority": "high",
                "category": "deployment",
                "title": "Do Not Deploy Without Audit",
                "description": "This model has a high risk score. Conduct a thorough "
                               "manual audit before deploying to any environment. Consider "
                               "retraining from a trusted source.",
                "action": "Engage your security team for a detailed review.",
            })

        # LSB-specific
        lsb_score = category_scores.get("lsb_detector", {}).get("raw_score", 0)
        if lsb_score > 30:
            recommendations.append({
                "priority": "medium",
                "category": "steganography",
                "title": "Apply Weight Quantization",
                "description": "LSB steganography indicators detected. Quantizing the model "
                               "weights to lower precision (e.g., INT8) can destroy hidden "
                               "data in the least significant bits without significantly "
                               "impacting model accuracy.",
                "action": "Apply post-training quantization (PTQ) and re-evaluate performance.",
            })

        # Backdoor-specific
        backdoor_score = category_scores.get("backdoor_detector", {}).get("raw_score", 0)
        if backdoor_score > 30:
            recommendations.append({
                "priority": "medium",
                "category": "backdoor",
                "title": "Run Neural Cleanse or Fine-Pruning",
                "description": "Backdoor indicators suggest potential trigger-based behavior. "
                               "Apply defense techniques such as Neural Cleanse, Fine-Pruning, "
                               "or Activation Clustering to identify and neutralize potential "
                               "backdoors.",
                "action": "Apply fine-pruning: prune neurons with abnormal activation patterns "
                          "and fine-tune on clean data.",
            })

        # Entropy-specific
        entropy_score = category_scores.get("entropy_analyzer", {}).get("raw_score", 0)
        if entropy_score > 30:
            recommendations.append({
                "priority": "medium",
                "category": "entropy",
                "title": "Verify Against Reference Model",
                "description": "Unusual entropy patterns detected. Compare this model's weight "
                               "distributions against a known-clean reference model of the same "
                               "architecture to isolate manipulated layers.",
                "action": "Obtain the original model from a trusted source and compare "
                          "layer-by-layer statistics.",
            })

        # General for any score
        if score <= 25:
            recommendations.append({
                "priority": "low",
                "category": "general",
                "title": "Standard Deployment Precautions",
                "description": "Model appears clean. Apply standard security practices: "
                               "use SafeTensors format, verify provenance hash, run in "
                               "sandboxed inference environment, and monitor production behavior.",
                "action": "Record the SHA-256 hash for provenance tracking.",
            })

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda r: priority_order.get(r["priority"], 3))

        return recommendations
