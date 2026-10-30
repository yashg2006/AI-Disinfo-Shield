"""
ModelSentinel — Demo Model Generator
Creates sample clean and tampered models for the hackathon demonstration.

Generates 4 models:
1. clean_cnn.pt — Standard CNN with normal weight distributions
2. stego_lsb_cnn.pt — CNN with a message hidden in LSBs of conv1 weights
3. backdoor_cnn.pt — CNN with injected outlier neurons simulating a backdoor
4. stego_structured_cnn.pt — CNN with periodic noise embedded as a covert channel

Also generates SafeTensors variants for format comparison testing.
"""

import os
import struct
import numpy as np
import torch
import torch.nn as nn

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "models")


class DemoCNN(nn.Module):
    """Simple CNN for demonstration purposes."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def create_realistic_weights(model: DemoCNN) -> DemoCNN:
    """
    Initialize the model with realistic-looking trained weights
    by running a few iterations of fake training.
    """
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Brief pseudo-training on random data to create realistic distributions
    for _ in range(50):
        fake_input = torch.randn(16, 3, 32, 32)
        fake_labels = torch.randint(0, 10, (16,))
        output = model(fake_input)
        loss = criterion(output, fake_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    return model


# ============================================================
# Attack 1: LSB Steganography
# ============================================================

def embed_lsb_message(weights: np.ndarray, message: str) -> np.ndarray:
    """
    Embed a text message into the least significant bits of float32 weight values.
    This replaces the LSB of each weight's IEEE 754 representation with a message bit.
    """
    # Convert message to binary
    message_bytes = message.encode('utf-8')
    # Add length header (4 bytes) + message + padding
    header = struct.pack('>I', len(message_bytes))
    payload = header + message_bytes

    # Convert to bit array
    bits = []
    for byte in payload:
        for bit_pos in range(8):
            bits.append((byte >> (7 - bit_pos)) & 1)

    # Embed bits into weight LSBs
    flat = weights.flatten().copy()
    raw_bytes = flat.astype(np.float32).tobytes()
    uint32_view = np.frombuffer(raw_bytes, dtype=np.uint32).copy()

    n_bits = min(len(bits), len(uint32_view))
    for i in range(n_bits):
        # Clear LSB, then set it to the message bit
        uint32_view[i] = (uint32_view[i] & 0xFFFFFFFE) | bits[i]

    # Convert back to float32
    modified = np.frombuffer(uint32_view.tobytes(), dtype=np.float32)
    return modified.reshape(weights.shape)


def create_stego_lsb_model() -> dict:
    """Create a model with a secret message hidden in conv1 weights via LSB steganography."""
    model = DemoCNN()
    model = create_realistic_weights(model)
    state = model.state_dict()

    # Embed message in conv1.weight
    secret_message = "MALWARE_PAYLOAD_EXFILTRATE_DATA_TO_C2_SERVER_192.168.1.100:8443"
    conv1_weights = state['conv1.weight'].numpy()
    modified_weights = embed_lsb_message(conv1_weights, secret_message)
    state['conv1.weight'] = torch.from_numpy(modified_weights)

    return state


# ============================================================
# Attack 2: Backdoor Injection
# ============================================================

def create_backdoor_model() -> dict:
    """
    Create a model with artificial backdoor indicators:
    - Inject outlier neurons in the FC layer
    - Create a bimodal weight distribution
    """
    model = DemoCNN()
    model = create_realistic_weights(model)
    state = model.state_dict()

    # Inject outlier weights in fc1 (simulating backdoor trigger neurons)
    fc1_weights = state['fc1.weight'].numpy()

    # Select 10% of neurons to be "backdoor neurons" with abnormally large weights
    n_neurons = fc1_weights.shape[0]
    n_backdoor = max(1, n_neurons // 10)
    backdoor_indices = np.random.choice(n_neurons, n_backdoor, replace=False)

    for idx in backdoor_indices:
        # Set these neurons to have very large magnitude weights
        fc1_weights[idx, :] *= 15.0  # 15x amplification
        # Also add some structured pattern (trigger signature)
        pattern = np.sin(np.linspace(0, 4 * np.pi, fc1_weights.shape[1])) * 5.0
        fc1_weights[idx, :] += pattern

    state['fc1.weight'] = torch.from_numpy(fc1_weights)

    # Also modify fc2 bias to create asymmetric activation for one class
    fc2_bias = state['fc2.bias'].numpy()
    fc2_bias[7] += 3.0  # Boost class 7 (simulating the backdoor target class)
    state['fc2.bias'] = torch.from_numpy(fc2_bias)

    return state


# ============================================================
# Attack 3: Structured Noise / Covert Channel
# ============================================================

def create_stego_structured_model() -> dict:
    """
    Create a model with structured periodic noise injected into weights,
    simulating a frequency-domain covert channel.
    """
    model = DemoCNN()
    model = create_realistic_weights(model)
    state = model.state_dict()

    # Inject periodic signal into conv2 weights
    conv2_weights = state['conv2.weight'].numpy().flatten()

    # Create a structured periodic signal (simulating encoded data)
    n = len(conv2_weights)
    frequencies = [0.01, 0.037, 0.073]  # Multiple carrier frequencies
    signal = np.zeros(n)
    for freq in frequencies:
        signal += 0.001 * np.sin(2 * np.pi * freq * np.arange(n))

    conv2_weights += signal.astype(np.float32)
    state['conv2.weight'] = torch.from_numpy(
        conv2_weights.reshape(state['conv2.weight'].shape)
    )

    # Also inject correlated noise between conv2 and conv3
    conv3_weights = state['conv3.weight'].numpy().flatten()
    # Make conv3 partially correlated with conv2 (unusual cross-layer dependency)
    min_len = min(len(conv2_weights), len(conv3_weights))
    conv3_weights[:min_len] += conv2_weights[:min_len] * 0.3
    state['conv3.weight'] = torch.from_numpy(
        conv3_weights.reshape(state['conv3.weight'].shape)
    )

    return state


# ============================================================
# Main Generator
# ============================================================

def generate_all():
    """Generate all demo models."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  ModelSentinel -- Demo Model Generator")
    print("=" * 60)
    print()

    # 1. Clean model
    print("[1/4] Generating clean_cnn.pt ...")
    model = DemoCNN()
    model = create_realistic_weights(model)
    clean_state = model.state_dict()
    torch.save(clean_state, os.path.join(OUTPUT_DIR, "clean_cnn.pt"))
    _save_safetensors(clean_state, os.path.join(OUTPUT_DIR, "clean_cnn.safetensors"))
    print("      [OK] Clean CNN -- normally trained, no manipulation")
    print()

    # 2. LSB Steganography model
    print("[2/4] Generating stego_lsb_cnn.pt ...")
    stego_state = create_stego_lsb_model()
    torch.save(stego_state, os.path.join(OUTPUT_DIR, "stego_lsb_cnn.pt"))
    _save_safetensors(stego_state, os.path.join(OUTPUT_DIR, "stego_lsb_cnn.safetensors"))
    print("      [!!] LSB Stego -- hidden message in conv1 weight LSBs")
    print()

    # 3. Backdoor model
    print("[3/4] Generating backdoor_cnn.pt ...")
    backdoor_state = create_backdoor_model()
    torch.save(backdoor_state, os.path.join(OUTPUT_DIR, "backdoor_cnn.pt"))
    _save_safetensors(backdoor_state, os.path.join(OUTPUT_DIR, "backdoor_cnn.safetensors"))
    print("      [XX] Backdoor -- outlier neurons in FC layer + biased output")
    print()

    # 4. Structured noise model
    print("[4/4] Generating stego_structured_cnn.pt ...")
    structured_state = create_stego_structured_model()
    torch.save(structured_state, os.path.join(OUTPUT_DIR, "stego_structured_cnn.pt"))
    _save_safetensors(structured_state, os.path.join(OUTPUT_DIR, "stego_structured_cnn.safetensors"))
    print("      [**] Structured -- periodic noise covert channel + cross-layer correlation")
    print()

    # Summary
    print("=" * 60)
    print(f"  Generated 8 model files in: {OUTPUT_DIR}")
    print()
    print("  PyTorch files (.pt):")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.pt'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"    - {f} ({size / 1024:.1f} KB)")
    print()
    print("  SafeTensors files (.safetensors):")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.safetensors'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"    • {f} ({size / 1024:.1f} KB)")
    print()
    print("  Ready for scanning with ModelSentinel!")
    print("=" * 60)


def _save_safetensors(state_dict: dict, path: str):
    """Save a state dict as SafeTensors format."""
    try:
        from safetensors.torch import save_file
        save_file(state_dict, path)
    except ImportError:
        print("      [i] safetensors not installed, skipping .safetensors variant")


if __name__ == "__main__":
    generate_all()
