"""Transcriptome adapter: projects pre-computed cell-foundation-model
embeddings into the LLM token embedding space via a linear adapter layer.

Users run their own cell foundation model (e.g. scGPT, Geneformer) **outside**
the training loop and store the resulting embedding vectors in the dataset.
This module only handles the final projection step.

Typical usage
-------------
>>> adapter = TranscriptomeAdapter(embedding_dim=1152, llm_hidden_dim=4096)
>>> embeds = adapter(torch.randn(2, 1152))  # (batch, num_tokens, llm_hidden_dim)
"""

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TranscriptomeAdapter(nn.Module):
    """Project pre-computed transcriptome embeddings into the LLM token
    embedding space.

    Parameters
    ----------
    embedding_dim : int
        Dimensionality of the pre-computed embeddings coming from the cell
        foundation model (default 1152, matching Geneformer V2).
    llm_hidden_dim : int
        Hidden dimension of the target LLM.
    num_tokens : int
        Number of virtual tokens produced per transcriptome sample.
    """

    def __init__(
        self,
        embedding_dim: int = 1152,
        llm_hidden_dim: int = 4096,
        num_tokens: int = 1,
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, llm_hidden_dim * num_tokens)
        self.num_tokens = num_tokens
        self.llm_hidden_dim = llm_hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(batch, embedding_dim)``
            Pre-computed embeddings from a cell foundation model.

        Returns
        -------
        Tensor of shape ``(batch, num_tokens, llm_hidden_dim)``
            Token embeddings ready to be consumed by the LLM.
        """
        projected = self.projection(x)  # (batch, llm_hidden_dim * num_tokens)
        return projected.view(-1, self.num_tokens, self.llm_hidden_dim)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def save_pretrained(self, path: str) -> None:
        """Save the adapter weights to *path*."""
        torch.save(self.state_dict(), path)
        logger.info("TranscriptomeAdapter saved to %s", path)

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: Any) -> "TranscriptomeAdapter":
        """Load a previously saved ``TranscriptomeAdapter`` from *path*.

        Parameters
        ----------
        path : str
            File path to a checkpoint saved with :meth:`save_pretrained`.
        **kwargs
            Forwarded to ``TranscriptomeAdapter.__init__``.
        """
        adapter = cls(**kwargs)
        state = torch.load(path, map_location="cpu", weights_only=True)
        adapter.load_state_dict(state)
        logger.info("TranscriptomeAdapter loaded from %s", path)
        return adapter
