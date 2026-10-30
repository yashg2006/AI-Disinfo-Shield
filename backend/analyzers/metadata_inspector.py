"""
ModelSentinel — Metadata & Format Inspector
Inspects model format, metadata, and pickle safety to identify
supply-chain risks and potential malicious code execution vectors.
"""

import os
import struct
from typing import Dict, List

from backend.model_parser import ModelArtifact

# Dangerous pickle opcodes / modules
DANGEROUS_MODULES = {
    "os", "subprocess", "sys", "shutil", "builtins",
    "socket", "http", "urllib", "requests", "ctypes",
    "importlib", "runpy", "code", "codeop", "compile",
    "eval", "exec", "execfile", "__import__",
    "nt", "posix", "posixpath", "ntpath",
    "tempfile", "glob", "pathlib",
    "pickle", "shelve", "marshal",
    "webbrowser", "antigravity",
}

DANGEROUS_FUNCTIONS = {
    "system", "popen", "exec", "eval", "execfile",
    "compile", "input", "__import__", "getattr", "setattr",
    "globals", "locals", "vars",
    "Popen", "call", "check_output", "check_call",
    "connect", "bind", "listen",
    "urlopen", "urlretrieve", "Request",
}


class MetadataInspector:
    """Inspect model format and metadata for safety concerns."""

    def analyze(self, artifact: ModelArtifact) -> Dict:
        """
        Run metadata and format inspection.

        Returns:
            Dict with 'format_info', 'pickle_scan', 'metadata_check', 'alerts', 'summary'.
        """
        alerts = []
        score = 0

        # --- Format safety assessment ---
        format_info = self._assess_format(artifact)
        if format_info.get("is_pickle_based", False):
            alerts.append({
                "severity": "warning",
                "type": "pickle_format",
                "message": f"Model uses pickle-based format ({artifact.model_format}). "
                           f"Pickle files can contain arbitrary Python code that executes "
                           f"on load. Consider converting to SafeTensors format for safe "
                           f"weight-only storage without code execution risk.",
                "value": artifact.model_format,
                "threshold": "safetensors",
            })
            score += 15

        # --- Pickle security scan ---
        pickle_scan = {"scanned": False}
        if artifact.model_format == "pytorch":
            pickle_scan = self._scan_pickle(artifact.file_path)
            if pickle_scan.get("dangerous_ops"):
                score += 50  # Major security issue
                for op in pickle_scan["dangerous_ops"]:
                    alerts.append({
                        "severity": "critical",
                        "type": "dangerous_pickle_op",
                        "message": f"CRITICAL: Detected dangerous pickle operation: {op['description']}. "
                                   f"Module '{op.get('module', 'unknown')}' with function "
                                   f"'{op.get('function', 'unknown')}' can execute arbitrary code "
                                   f"when the model is loaded. This is a strong indicator of "
                                   f"a malicious model file.",
                        "value": op["description"],
                        "threshold": "none",
                    })

        # --- State dict key analysis ---
        metadata_check = self._check_metadata(artifact)
        if metadata_check.get("suspicious_keys"):
            score += 10
            for key in metadata_check["suspicious_keys"]:
                alerts.append({
                    "severity": "warning",
                    "type": "suspicious_state_dict_key",
                    "message": f"State dict contains unusual key '{key}' that does not "
                               f"follow standard layer naming conventions. Non-standard keys "
                               f"may be used to store hidden configuration or payload metadata.",
                    "value": key,
                    "threshold": "standard naming",
                })

        if metadata_check.get("has_non_tensor_values", False):
            alerts.append({
                "severity": "info",
                "type": "non_tensor_values",
                "message": "State dict contains non-tensor values. While this can be "
                           "legitimate (e.g., buffer counts, config), it increases the "
                           "attack surface for embedded payloads.",
                "value": True,
                "threshold": False,
            })

        # --- File integrity ---
        integrity = {
            "sha256": artifact.sha256,
            "file_size_bytes": artifact.file_size_bytes,
            "file_size_human": self._human_size(artifact.file_size_bytes),
        }

        return {
            "analyzer": "metadata_inspector",
            "format_info": format_info,
            "pickle_scan": pickle_scan,
            "metadata_check": metadata_check,
            "integrity": integrity,
            "alerts": alerts,
            "summary": {
                "is_pickle_based": format_info.get("is_pickle_based", False),
                "dangerous_ops_found": len(pickle_scan.get("dangerous_ops", [])),
                "suspicious_keys_found": len(metadata_check.get("suspicious_keys", [])),
                "score": min(100, score),
            },
        }

    def _assess_format(self, artifact: ModelArtifact) -> Dict:
        """Assess the safety of the model file format."""
        is_pickle = artifact.model_format in ("pytorch",)
        is_safe = artifact.model_format == "safetensors"

        return {
            "format": artifact.model_format,
            "is_pickle_based": is_pickle,
            "is_safe_format": is_safe,
            "recommendation": (
                "✅ SafeTensors format — no code execution risk"
                if is_safe
                else "⚠️ Pickle-based format — convert to SafeTensors to eliminate "
                     "arbitrary code execution risk"
            ),
        }

    def _scan_pickle(self, file_path: str) -> Dict:
        """
        Scan a pickle file for dangerous opcodes using fickling or token extraction.
        """
        dangerous_ops = []

        try:
            from fickling.fickle import Fickling
            fickled = Fickling.load(file_path)

            if hasattr(fickled, 'properties') and hasattr(fickled.properties, 'imports'):
                for imp in fickled.properties.imports:
                    module_str = str(imp).strip()
                    top_mod = module_str.split('.')[0]
                    if top_mod in DANGEROUS_MODULES:
                        dangerous_ops.append({
                            "type": "import",
                            "module": module_str,
                            "function": "",
                            "description": f"Dangerous module import: {module_str}",
                        })

            return {
                "scanned": True,
                "scanner": "fickling",
                "is_safe": len(dangerous_ops) == 0,
                "dangerous_ops": dangerous_ops,
            }
        except Exception:
            pass

        # Fallback: exact opcode & token scan
        try:
            dangerous_ops = self._manual_pickle_scan(file_path)
        except Exception:
            pass

        return {
            "scanned": True,
            "scanner": "manual",
            "is_safe": len(dangerous_ops) == 0,
            "dangerous_ops": dangerous_ops,
        }

    def _manual_pickle_scan(self, file_path: str) -> List[Dict]:
        """Scan pickle file byte stream for exact dangerous module and function calls."""
        import re
        dangerous_ops = []

        try:
            with open(file_path, "rb") as f:
                content = f.read()

            # Extract printable ASCII tokens delimited by non-alphanumeric chars
            tokens = set(t.decode('ascii', errors='ignore') for t in re.findall(rb'[a-zA-Z_][a-zA-Z0-9_\.]*', content))

            for mod in DANGEROUS_MODULES:
                if mod in tokens:
                    # Check if token is standalone module reference
                    dangerous_ops.append({
                        "type": "module_reference",
                        "module": mod,
                        "function": "",
                        "description": f"Pickle references dangerous module: {mod}",
                    })

            for func in DANGEROUS_FUNCTIONS:
                if func in tokens:
                    dangerous_ops.append({
                        "type": "function_reference",
                        "module": "",
                        "function": func,
                        "description": f"Pickle references dangerous function: {func}",
                    })
        except Exception:
            pass

        return dangerous_ops


    def _check_metadata(self, artifact: ModelArtifact) -> Dict:
        """Check state dict keys for suspicious patterns."""
        suspicious_keys = []
        standard_prefixes = {
            "weight", "bias", "running_mean", "running_var",
            "num_batches_tracked", "bn", "conv", "fc", "linear",
            "layer", "encoder", "decoder", "head", "classifier",
            "embedding", "norm", "attention", "transformer",
        }

        for key in artifact.raw_state_dict_keys:
            parts = key.lower().replace(".", "_").split("_")
            # Check if any part of the key matches standard naming
            is_standard = any(
                prefix in part
                for part in parts
                for prefix in standard_prefixes
            )
            # Also check for numeric-only parts (layer indices)
            has_only_weird = all(
                not any(prefix in part for prefix in standard_prefixes)
                and not part.isdigit()
                for part in parts
                if len(part) > 0
            )

            if has_only_weird and len(key) > 3:
                suspicious_keys.append(key)

        return {
            "total_keys": len(artifact.raw_state_dict_keys),
            "suspicious_keys": suspicious_keys[:10],  # Limit to top 10
            "has_non_tensor_values": artifact.has_non_tensor_values,
        }

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable size string."""
        for unit in ("B", "KB", "MB", "GB"):
            if abs(size_bytes) < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
