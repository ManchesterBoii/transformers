import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    Simplified Vector Quantization layer for VQ-ViTMAE, adapted from lucidrains implementation.
    """

    def __init__(self, config):
        super(VectorQuantizer, self).__init__()
        self.codebook_size = config.num_codebooks
        self.embedding_dim = config.embedding_dim
        self.commitment_cost = config.commitment_cost
        # Update config to include a codebook cost term
        self.codebook_cost = config.codebook_cost

        # Initialize the codebook
        self.codebook = nn.Parameter(torch.randn(self.codebook_size, self.embedding_dim))

        # Optional quantization dropout
        self.quantization_dropout = (
            nn.Dropout(config.quantization_dropout) if config.quantization_dropout > 0 else None
        )

    def forward(self, x, hidden_dim):
        """
        Args:
            x: Input tensor of shape `(batch_size, seq_length, embedding_dim)`
            (seq_length determined from number of unmasked patches)

        Returns:
            quantized: Tensor of quantized values replacing the inputs.
            commitment_loss: The commitment loss for optimization.
        """
        batch_size, seq_length, hidden_dim = x.size()
        if hidden_dim != self.embedding_dim:
            raise ValueError(
                f"Input embedding dimension ({hidden_dim}) does not match codebook embedding dimension ({self.embedding_dim})."
            )

        # Flatten input to (batch_size * seq_length, embedding_dim)
        flat_input = x.view(-1, hidden_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            + torch.sum(self.codebook**2, dim=1)
            - 2 * torch.matmul(flat_input, self.codebook.t())
        )

        # Find the closest codebook entry for each input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.codebook[encoding_indices.squeeze()].view_as(x)

        # Apply dropout to quantized values if specified
        if self.quantization_dropout:
            quantized = self.quantization_dropout(quantized)

        # Compute losses
        codebook_loss = F.mse_loss(quantized, x.detach()) 
        commitment_loss = F.mse_loss(quantized.detach(), x)
        total_loss = self.codebook_cost * codebook_loss + self.commitment_cost * commitment_loss

        # Use straight-through estimator to pass gradients to the encoder
        quantized = x + (quantized - x).detach()

        return quantized, total_loss