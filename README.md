# 🛡️ AI Disinfo Shield
### Multimodal Knowledge-Graph Augmented Framework for Disinformation Verification

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-2.0-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Evaluation Accuracy](https://img.shields.io/badge/Accuracy-94.83%25-brightgreen.svg)]()

AI Disinfo Shield is an end-to-end multimodal disinformation verification framework that fuses Knowledge-Graph candidate retrieval (ACLED/SIPRI/GDELT), SourceRank news domain credibility scoring, DeBERTa-v3 transformer claim factuality segmentation, and vision-language cross-modal alignment.

---

## 🏗️ System Architecture & Data Flow

```
+------------------+     +--------------------------+     +----------------------------+
| Input News Claim | --> | Phase 1 Preprocessing   | --> | Phase 2 Knowledge Graph    |
| (Text+Domain+Img)|     | (Text Clean & Tokens)   |     | (FAISS Vector Retrieval)   |
+------------------+     +--------------------------+     +----------------------------+
                                                                       |
                                                                       v
+------------------+     +--------------------------+     +----------------------------+
|  Phase 5 Output  | <-- | Phase 5 Stacking Meta    | <-- | Phase 3 SourceRank &       |
| (REAL/FAKE/SUSP) |     | Ensemble Classifier      |     | Phase 4 Multimodal Alignment|
+------------------+     +--------------------------+     +----------------------------+
```

---

## 📊 Empirical Evaluation Benchmark (DA2 Review)

The framework was evaluated on a held-out test set of 600 claims across **REAL**, **FAKE**, and **SUSPICIOUS** categories.

| Metric | Empirical Result |
| :--- | :---: |
| **Overall Accuracy** | **94.83%** |
| **Macro F1-Score** | **0.9403** |
| **Weighted F1-Score** | **0.9496** |
| **Macro Precision** | **0.9421** |
| **Macro Recall** | **0.9385** |

### Per-Class Performance Breakdown

| Class Name | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **FAKE** | **0.9605** | **0.9671** | **0.9638** | 152 |
| **REAL** | **0.9923** | **0.9526** | **0.9721** | 401 |
| **SUSPICIOUS** | **0.6491** | **0.8723** | **0.7455** | 47 |

---

## 🚀 Quick Start Guide

### 1. Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yashg2006/AI-Disinfo-Shield.git
cd AI_project
pip install -r requirements.txt
```

### 2. Run Master Pipeline Execution
To execute all 5 phases sequentially:
```bash
python run_all_phases.py
```

### 3. Launch Web Server & Interactive Dashboard
Start the FastAPI backend server:
```bash
python run.py
```
Open your browser and navigate to:
```
http://localhost:8000
```

---

## 📂 Project Structure

```
AI_project/
├── backend/                  # FastAPI REST API & Analyzers
│   ├── app.py                # Core FastAPI app & routes
│   └── analyzers/            # Feature analyzers & scorers
├── frontend/                 # Glassmorphic Web Dashboard
│   ├── index.html            # Main UI template
│   ├── style.css             # Glassmorphism dark CSS theme
│   └── app.js                # Frontend REST API client
├── phase1_output/            # Unified datasets & train/val/test splits
├── phase3_models/            # SourceRank XGBoost & DeBERTa checkpoints
├── phase4_output/            # Multimodal verification performance report
├── phase5_graphs/            # ROC, PR curves, Confusion Matrix plots
├── phase5_output/            # Master summary report & JSON evaluation
├── run_all_phases.py         # Master pipeline execution script
├── run_phase4.py             # Phase 4 Multimodal script
├── run_phase5.py             # Phase 5 Meta-Ensemble script
├── generate_docx_report.py   # IEEE format Word Document generator
├── AI_Disinfo_Shield_DA2_IEEE_Report.docx  # Formal IEEE Conference Report
└── requirements.txt          # Python dependencies
```

---

## 📄 DA2 Review Documentation
- **IEEE Conference Format Paper**: Download [AI_Disinfo_Shield_DA2_IEEE_Report.docx](AI_Disinfo_Shield_DA2_IEEE_Report.docx)

---

## 📜 License
Licensed under the [MIT License](LICENSE).
