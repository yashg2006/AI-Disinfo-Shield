"""Quick test script for ModelSentinel API."""
import requests
import json
import sys
import os

BASE_URL = "http://localhost:8000"
MODELS_DIR = os.path.join(os.path.dirname(__file__), "demo", "models")


def test_scan(filename, expected_risk):
    filepath = os.path.join(MODELS_DIR, filename)
    print(f"\n{'='*60}")
    print(f"  Scanning: {filename}")
    print(f"  Expected: {expected_risk}")
    print(f"{'='*60}")

    with open(filepath, "rb") as f:
        response = requests.post(
            f"{BASE_URL}/api/scan",
            files={"file": (filename, f)},
        )

    if response.status_code != 200:
        print(f"  ERROR: HTTP {response.status_code}")
        print(f"  {response.text}")
        return False

    data = response.json()
    score = data["risk_score"]
    level = data["risk_level"]["label"]
    alerts = data["alert_counts"]

    print(f"  Risk Score: {score}/100")
    print(f"  Risk Level: {level}")
    print(f"  Alerts: {alerts['critical']} critical, {alerts['warning']} warnings, {alerts['info']} info")
    print(f"  Scan ID: {data['scan_id']}")

    # Show top alerts
    for alert in data["alerts"][:3]:
        print(f"  -> [{alert['severity'].upper()}] {alert['type']}: {alert['message'][:100]}...")

    return True


def main():
    # Health check
    try:
        r = requests.get(f"{BASE_URL}/api/health")
        print(f"Health check: {r.json()}")
    except Exception as e:
        print(f"Server not running! Start with: python run.py")
        sys.exit(1)

    # Test all models
    tests = [
        ("clean_cnn.pt", "LOW risk"),
        ("stego_lsb_cnn.pt", "HIGH risk (LSB steganography)"),
        ("backdoor_cnn.pt", "HIGH/CRITICAL risk (backdoor)"),
        ("stego_structured_cnn.pt", "HIGH risk (structured noise)"),
    ]

    for filename, expected in tests:
        test_scan(filename, expected)

    # Also test a safetensors variant
    test_scan("clean_cnn.safetensors", "LOW risk (safe format)")

    print(f"\n{'='*60}")
    print("  All tests complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
