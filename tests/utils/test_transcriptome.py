"""Unit tests for transcriptome multimodal support."""

import base64
import json

import torch

from slime.utils.processing_utils import encode_transcriptome_for_rollout_engine, process_transcriptome_info
from slime.utils.transcriptome import TranscriptomeEncoder
from slime.utils.types import MultimodalTypes


# ---------------------------------------------------------------------------
# MultimodalTypes
# ---------------------------------------------------------------------------
def test_transcriptome_type_registered():
    """TRANSCRIPTOME should be listed among all multimodal types."""
    assert MultimodalTypes.TRANSCRIPTOME is not None
    assert MultimodalTypes.TRANSCRIPTOME.name == "transcriptome"
    assert MultimodalTypes.TRANSCRIPTOME.placeholder == "<transcriptome>"
    assert MultimodalTypes.TRANSCRIPTOME in MultimodalTypes.all()


def test_multimodal_types_get_transcriptome():
    mt = MultimodalTypes.get("transcriptome")
    assert mt is not None
    assert mt.name == "transcriptome"


# ---------------------------------------------------------------------------
# process_transcriptome_info
# ---------------------------------------------------------------------------
def test_process_transcriptome_info_extracts_vectors():
    """process_transcriptome_info should collect vectors from content items."""
    vec1 = [0.1] * 100
    vec2 = [0.2] * 100
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "transcriptome", "transcriptome": vec1},
                {"type": "text", "text": "What cell type is this?"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "transcriptome", "transcriptome": vec2},
            ],
        },
    ]
    result = process_transcriptome_info(messages)
    assert "transcriptomes" in result
    assert len(result["transcriptomes"]) == 2
    assert result["transcriptomes"][0] == vec1
    assert result["transcriptomes"][1] == vec2


def test_process_transcriptome_info_no_transcriptome():
    """Should return an empty list when no transcriptome items are present."""
    messages = [
        {"role": "user", "content": "Hello, no transcriptome here."},
    ]
    result = process_transcriptome_info(messages)
    assert result["transcriptomes"] == []


def test_process_transcriptome_info_numpy_array():
    """Should handle numpy arrays gracefully."""
    import numpy as np

    vec = np.random.rand(100).astype(np.float32)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "transcriptome", "transcriptome": vec},
            ],
        },
    ]
    result = process_transcriptome_info(messages)
    assert len(result["transcriptomes"]) == 1
    assert isinstance(result["transcriptomes"][0], list)


# ---------------------------------------------------------------------------
# encode_transcriptome_for_rollout_engine
# ---------------------------------------------------------------------------
def test_encode_transcriptome_for_rollout_engine():
    """Encoded payload should round-trip back to the original vector."""
    vec = [0.5, 1.0, -0.3]
    encoded = encode_transcriptome_for_rollout_engine(vec)
    assert encoded.startswith("data:application/json;base64,")
    b64_part = encoded.split(",", 1)[1]
    decoded = json.loads(base64.b64decode(b64_part).decode("utf-8"))
    assert decoded == vec


def test_encode_transcriptome_numpy():
    """Should also accept numpy arrays."""
    import numpy as np

    vec = np.array([1.0, 2.0, 3.0])
    encoded = encode_transcriptome_for_rollout_engine(vec)
    b64_part = encoded.split(",", 1)[1]
    decoded = json.loads(base64.b64decode(b64_part).decode("utf-8"))
    assert decoded == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# TranscriptomeEncoder
# ---------------------------------------------------------------------------
def test_transcriptome_encoder_output_shape():
    """Encoder should produce (batch, num_tokens, llm_hidden_dim)."""
    encoder = TranscriptomeEncoder(input_dim=200, foundation_dim=64, llm_hidden_dim=128, num_tokens=1)
    x = torch.randn(3, 200)
    out = encoder(x)
    assert out.shape == (3, 1, 128)


def test_transcriptome_encoder_multiple_tokens():
    """When num_tokens > 1 the output should have that many token vectors."""
    encoder = TranscriptomeEncoder(input_dim=200, foundation_dim=64, llm_hidden_dim=128, num_tokens=4)
    x = torch.randn(2, 200)
    out = encoder(x)
    assert out.shape == (2, 4, 128)


def test_transcriptome_encoder_gradient_flow():
    """Gradients should flow from loss back to the encoder parameters."""
    encoder = TranscriptomeEncoder(input_dim=200, foundation_dim=64, llm_hidden_dim=128)
    x = torch.randn(2, 200)
    out = encoder(x)
    loss = out.sum()
    loss.backward()
    for p in encoder.parameters():
        assert p.grad is not None
