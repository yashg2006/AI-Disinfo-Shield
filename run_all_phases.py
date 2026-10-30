"""
AI Disinfo Shield -- Master Pipeline Runner (Phases 1 to 5)
Executes all 5 phases sequentially with dataset cutting for optimal execution speed.
Generates comprehensive benchmark graphs and master evaluation report.
"""
import os
import sys
import time

def main():
    start_time = time.time()
    print("=================================================================")
    print("  AI DISINFO SHIELD -- MASTER PIPELINE RUNNER (PHASE 1 TO 5)")
    print("=================================================================\n")

    print("[RUNNING PHASE 1] Data Ingestion, Cleaning & Stratified Splitting...")
    import phase1
    phase1.main()

    print("\n[RUNNING PHASE 2] Knowledge Graph Construction & Vector Retrieval...")
    import phase2
    phase2.main()

    print("\n[RUNNING PHASE 3A] SourceRank XGBoost Credibility Scoring...")
    import run_phase3a
    run_phase3a.main()

    print("\n[RUNNING PHASE 3B] DeBERTa Claim Segmentation & Factuality...")
    import run_phase3b
    run_phase3b.main()

    print("\n[RUNNING PHASE 4] Multimodal & Cross-Modal Verification...")
    import run_phase4
    run_phase4.main()

    print("\n[RUNNING PHASE 5] Meta-Ensemble Stacking & Master Graph Suite...")
    import run_phase5
    run_phase5.main()

    elapsed = time.time() - start_time
    print(f"\n=================================================================")
    print(f"  ALL 5 PHASES COMPLETED SUCCESSFULLY IN {elapsed:.2f} SECONDS!")
    print(f"=================================================================")

if __name__ == "__main__":
    main()
