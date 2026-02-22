"""Unit tests for transcriptome multimodal support (pre-computed embeddings)."""

import os
import tempfile

import torch

from slime.utils.processing_utils import process_transcriptome_info
from slime.utils.transcriptome import TranscriptomeAdapter
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
def test_process_transcriptome_info_extracts_embeddings():
    """process_transcriptome_info should collect embedding vectors from content items."""
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
    assert "transcriptome_embeddings" in result
    assert len(result["transcriptome_embeddings"]) == 2
    assert result["transcriptome_embeddings"][0] == vec1
    assert result["transcriptome_embeddings"][1] == vec2


def test_process_transcriptome_info_no_transcriptome():
    """Should return an empty list when no transcriptome items are present."""
    messages = [
        {"role": "user", "content": "Hello, no transcriptome here."},
    ]
    result = process_transcriptome_info(messages)
    assert result["transcriptome_embeddings"] == []


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
    assert len(result["transcriptome_embeddings"]) == 1
    assert isinstance(result["transcriptome_embeddings"][0], list)


# ---------------------------------------------------------------------------
# TranscriptomeAdapter
# ---------------------------------------------------------------------------
def test_adapter_output_shape():
    """Adapter should produce (batch, num_tokens, llm_hidden_dim)."""
    adapter = TranscriptomeAdapter(embedding_dim=1152, llm_hidden_dim=128, num_tokens=1)
    x = torch.randn(3, 1152)
    out = adapter(x)
    assert out.shape == (3, 1, 128)


def test_adapter_multiple_tokens():
    """When num_tokens > 1 the output should have that many token vectors."""
    adapter = TranscriptomeAdapter(embedding_dim=64, llm_hidden_dim=128, num_tokens=4)
    x = torch.randn(2, 64)
    out = adapter(x)
    assert out.shape == (2, 4, 128)


def test_adapter_gradient_flow():
    """Gradients should flow from loss back to the adapter parameters."""
    adapter = TranscriptomeAdapter(embedding_dim=64, llm_hidden_dim=128)
    x = torch.randn(2, 64)
    out = adapter(x)
    loss = out.sum()
    loss.backward()
    for p in adapter.parameters():
        assert p.grad is not None


def test_adapter_save_load_roundtrip():
    """save_pretrained -> from_pretrained should restore identical weights."""
    adapter = TranscriptomeAdapter(embedding_dim=64, llm_hidden_dim=128, num_tokens=2)
    x = torch.randn(2, 64)
    original_out = adapter(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "adapter.pt")
        adapter.save_pretrained(path)

        loaded = TranscriptomeAdapter.from_pretrained(path, embedding_dim=64, llm_hidden_dim=128, num_tokens=2)

    loaded_out = loaded(x)
    assert torch.allclose(original_out, loaded_out)
