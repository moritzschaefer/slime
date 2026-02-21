"""Transcriptome encoder: maps raw gene-expression vectors into LLM token
embedding space via a cell foundation model backbone and an adapter layer.

Typical usage
-------------
>>> encoder = TranscriptomeEncoder(input_dim=20000, foundation_dim=512, llm_hidden_dim=4096)
>>> embeds = encoder(torch.randn(2, 20000))  # (batch, num_tokens, llm_hidden_dim)

Custom foundation model
-----------------------
>>> my_model = MyScGPTEncoder()  # any nn.Module: (batch, input_dim) -> (batch, foundation_dim)
>>> encoder = TranscriptomeEncoder(
...     foundation_model=my_model,
...     foundation_dim=512,
...     llm_hidden_dim=4096,
... )

Loading from checkpoint
-----------------------
>>> encoder = TranscriptomeEncoder.from_pretrained("path/to/encoder.pt")
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TranscriptomeEncoder(nn.Module):
    """Encode 20,000-dimensional transcriptome vectors into the LLM token
    embedding space.

    The pipeline mirrors the vision-encoder + adapter pattern used in
    vision-language models:

    1. **Foundation model** – either a user-supplied ``nn.Module`` or a
       built-in feed-forward backbone (placeholder for a pre-trained cell
       foundation model such as scGPT or Geneformer).
    2. **Adapter** – a linear projection that maps the foundation-model
       embeddings to ``num_tokens`` vectors in the LLM's hidden dimension.

    Parameters
    ----------
    input_dim : int
        Dimensionality of raw gene-expression vectors (default 20,000).
        Ignored when *foundation_model* is provided.
    foundation_dim : int
        Output dimension of the cell foundation model backbone.
    llm_hidden_dim : int
        Hidden dimension of the target LLM.
    num_tokens : int
        Number of virtual tokens produced per transcriptome sample.
    foundation_model : nn.Module | None
        An optional pre-trained cell foundation model.  Must accept an input
        tensor of shape ``(batch, input_dim)`` and return a tensor of shape
        ``(batch, foundation_dim)``.  When *None* (default) a built-in
        feed-forward network is used.
    """

    def __init__(
        self,
        input_dim: int = 20000,
        foundation_dim: int = 512,
        llm_hidden_dim: int = 4096,
        num_tokens: int = 1,
        foundation_model: nn.Module | None = None,
    ):
        super().__init__()
        if foundation_model is not None:
            self.foundation_model = foundation_model
        else:
            # Built-in placeholder backbone
            self.foundation_model = nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.GELU(),
                nn.Linear(2048, foundation_dim),
                nn.LayerNorm(foundation_dim),
            )
        # Adapter projects into LLM token embedding space
        self.adapter = nn.Linear(foundation_dim, llm_hidden_dim * num_tokens)
        self.num_tokens = num_tokens
        self.llm_hidden_dim = llm_hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(batch, input_dim)``
            Raw gene-expression vectors.

        Returns
        -------
        Tensor of shape ``(batch, num_tokens, llm_hidden_dim)``
            Token embeddings ready to be consumed by the LLM.
        """
        embeddings = self.foundation_model(x)  # (batch, foundation_dim)
        projected = self.adapter(embeddings)  # (batch, llm_hidden_dim * num_tokens)
        return projected.view(-1, self.num_tokens, self.llm_hidden_dim)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_pretrained(self, path: str) -> None:
        """Save the full encoder (foundation model + adapter) to *path*."""
        torch.save(self.state_dict(), path)
        logger.info("TranscriptomeEncoder saved to %s", path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "TranscriptomeEncoder":
        """Load a previously saved ``TranscriptomeEncoder`` from *path*.

        Any extra *kwargs* are forwarded to the constructor so that
        architecture hyper-parameters (``foundation_dim``, ``llm_hidden_dim``,
        ``num_tokens``, ``foundation_model``, …) can be overridden.

        Parameters
        ----------
        path : str
            File path to a checkpoint saved with :meth:`save_pretrained`.
        **kwargs
            Forwarded to ``TranscriptomeEncoder.__init__``.
        """
        encoder = cls(**kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state)
        logger.info("TranscriptomeEncoder loaded from %s", path)
        return encoder
