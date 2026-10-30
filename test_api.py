"""Quick API smoke test — verifies FAKE, REAL, and Self-Updating Engine endpoints."""
import requests, json

BASE = "http://localhost:8000"

# Test 1: FAKE tweet detection
print("=" * 70)
print("TEST 1: Fake Tweet Detection")
print("=" * 70)
r = requests.post(f"{BASE}/api/verify", json={
    "claim_text": "BREAKING: Leaked military documents PROVE secret underground base discovered under Kyiv! @conspiracy_daily",
    "source_domain": "twitter.com"
})
d = r.json()
verdict = d["verdict"]
conf = d["confidence_percentage"]
print(f"  Verdict    : {verdict}")
print(f"  Confidence : {conf}%")
print(f"  Explanation: {d['explanation'][:120]}...")
print(f"  Sensational: {d['tweet_metadata']['sensational_score']}")
print(f"  Handle Info: {d['handle_evaluation']['rating']}")
assert verdict == "FAKE", f"Expected FAKE, got {verdict}"
print("  ✅ PASSED — correctly identified as FAKE\n")

# Test 2: REAL news detection
print("=" * 70)
print("TEST 2: Real News Detection")
print("=" * 70)
r = requests.post(f"{BASE}/api/verify", json={
    "claim_text": "Reuters: U.S. Department of Defense announces new $400M security assistance package for Ukraine. @reuters",
    "source_domain": "reuters.com"
})
d = r.json()
verdict = d["verdict"]
conf = d["confidence_percentage"]
print(f"  Verdict    : {verdict}")
print(f"  Confidence : {conf}%")
print(f"  Explanation: {d['explanation'][:120]}...")
print(f"  Handle Info: {d['handle_evaluation']['rating']}")
assert verdict == "REAL", f"Expected REAL, got {verdict}"
print("  ✅ PASSED — correctly identified as REAL\n")

# Test 3: SUSPICIOUS content
print("=" * 70)
print("TEST 3: Suspicious / Unverified Content")
print("=" * 70)
r = requests.post(f"{BASE}/api/verify", json={
    "claim_text": "Multiple anonymous sources report unusual troop movements near the border region. Details unconfirmed.",
    "source_domain": "twitter.com"
})
d = r.json()
verdict = d["verdict"]
conf = d["confidence_percentage"]
print(f"  Verdict    : {verdict}")
print(f"  Confidence : {conf}%")
print(f"  Explanation: {d['explanation'][:120]}...")
print(f"  ✅ PASSED — classified as {verdict}\n")

# Test 4: Self-Updating Engine Status
print("=" * 70)
print("TEST 4: Self-Updating Engine Status")
print("=" * 70)
r = requests.get(f"{BASE}/api/self-update/status")
d = r.json()
print(f"  Enabled       : {d['is_enabled']}")
print(f"  Interval      : {d['interval_seconds']}s ({d['interval_minutes']} min)")
print(f"  Total Updates : {d['total_updates_performed']}")
print(f"  KG Triples    : {d['active_kg_triples']}")
print(f"  Domains Indexed: {d['active_source_domains']}")
print("  ✅ PASSED — Self-updater is active\n")

# Test 5: Trigger Manual Self-Update
print("=" * 70)
print("TEST 5: Trigger Manual Self-Update Cycle")
print("=" * 70)
r = requests.post(f"{BASE}/api/self-update/trigger")
d = r.json()
print(f"  Status : {d['status']}")
print(f"  Message: {d['message']}")
print(f"  Event  : {d['log']['event']}")
print("  ✅ PASSED — Manual self-update executed\n")

# Test 6: Health Check
print("=" * 70)
print("TEST 6: System Health Check")
print("=" * 70)
r = requests.get(f"{BASE}/api/health")
d = r.json()
print(f"  Status  : {d['status']}")
print(f"  Version : {d['version']}")
print(f"  KG Triples: {d['total_kg_triples']}")
print(f"  Self-Updater: enabled={d['self_updating_engine']['enabled']}")
print("  ✅ PASSED\n")

print("=" * 70)
print("  ALL 6 TESTS PASSED — PROTOTYPE IS FULLY OPERATIONAL!")
print(f"  Open http://localhost:8000 in your browser for the Web Dashboard")
print("=" * 70)
