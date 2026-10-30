# ModelSentinel — Demo Models

This directory contains sample AI models for demonstrating ModelSentinel's detection capabilities.

## How to Generate

```bash
cd AI_project
python demo/generate_demo_models.py
```

This creates 8 files (4 PyTorch + 4 SafeTensors) in `demo/models/`.

## Model Descriptions

### 1. `clean_cnn` — ✅ Clean Model (Expected: Low Risk)
- **Architecture**: Simple CNN (3×Conv2D + 2×FC, ~600K params)
- **Status**: Normally trained on random data, no manipulation
- **Expected detection**: No alerts. Risk score < 25.

### 2. `stego_lsb_cnn` — ⚠️ LSB Steganography (Expected: High Risk)
- **Architecture**: Same CNN
- **Attack**: A secret message `"MALWARE_PAYLOAD_EXFILTRATE_DATA_TO_C2_SERVER_192.168.1.100:8443"` is hidden in the least significant bits of `conv1.weight` using IEEE 754 bit manipulation.
- **Expected detection**:
  - LSB Detector should flag `conv1.weight` with high chi-squared p-value (uniform LSBs)
  - Low autocorrelation in LSB sequence
  - RS analysis should show R ≈ S

### 3. `backdoor_cnn` — 🔴 Backdoor Injection (Expected: Critical Risk)
- **Architecture**: Same CNN
- **Attack**: 10% of neurons in `fc1.weight` are amplified by 15× and injected with a sinusoidal trigger pattern. `fc2.bias[7]` is boosted to bias the model toward class 7 on trigger activation.
- **Expected detection**:
  - Backdoor Detector should flag neuron outliers (>3σ)
  - Statistical Profiler should detect high kurtosis in `fc1.weight`
  - Bimodality check should trigger

### 4. `stego_structured_cnn` — 🔶 Structured Noise Covert Channel (Expected: High Risk)
- **Architecture**: Same CNN
- **Attack**: Multi-frequency periodic signals injected into `conv2.weight` (simulating frequency-domain data encoding). Additionally, `conv3.weight` is made partially correlated with `conv2.weight` (cross-layer dependency anomaly).
- **Expected detection**:
  - Layer Correlation Analyzer should flag cross-layer correlation between conv2 and conv3
  - Spectral analysis should detect periodic peaks in conv2 FFT
  - Entropy may show slight anomalies

## Threat Model Summary

| Model | Attack Type | Target Layer | Detection Module |
|---|---|---|---|
| clean_cnn | None | — | — |
| stego_lsb_cnn | LSB Steganography | conv1.weight | LSB Detector |
| backdoor_cnn | Backdoor Injection | fc1.weight, fc2.bias | Backdoor Detector, Statistical |
| stego_structured_cnn | Covert Channel | conv2.weight, conv3.weight | Layer Correlation, Spectral |
