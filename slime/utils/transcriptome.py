"""Transcriptome encoder: maps raw gene-expression vectors into LLM token
embedding space via a cell foundation model backbone and an adapter layer.

Typical usage
-------------
>>> encoder = TranscriptomeEncoder(input_dim=20000, foundation_dim=512, llm_hidden_dim=4096)
>>> embeds = encoder(torch.randn(2, 20000))  # (batch, num_tokens, llm_hidden_dim)
"""

import torch
import torch.nn as nn


class TranscriptomeEncoder(nn.Module):
    """Encode 20 000-dimensional transcriptome vectors into the LLM token
    embedding space.

    The pipeline mirrors the vision-encoder + adapter pattern used in
    vision-language models:

    1. **Foundation model** – a small feed-forward backbone (placeholder for a
       pre-trained cell foundation model such as scGPT or Geneformer).
    2. **Adapter** – a linear projection that maps the foundation-model
       embeddings to ``num_tokens`` vectors in the LLM's hidden dimension.

    Parameters
    ----------
    input_dim : int
        Dimensionality of raw gene-expression vectors (default 20 000).
    foundation_dim : int
        Output dimension of the cell foundation model backbone.
    llm_hidden_dim : int
        Hidden dimension of the target LLM.
    num_tokens : int
        Number of virtual tokens produced per transcriptome sample.
    """

    def __init__(
        self,
        input_dim: int = 20000,
        foundation_dim: int = 512,
        llm_hidden_dim: int = 4096,
        num_tokens: int = 1,
    ):
        super().__init__()
        # Cell foundation model backbone (replace with a pre-trained model as needed)
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
