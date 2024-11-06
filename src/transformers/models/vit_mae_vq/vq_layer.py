import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    """
    Implements the vector quantization layer for VQ-ViTMAE.
    """

    def __init__(self, config):
        super(VectorQuantizer, self).__init__()
        self.num_codebooks = config.num_codebooks
        self.embedding_dim = config.embedding_dim
        self.commitment_cost = config.commitment_cost

        # Codebook embeddings
        self.codebook = nn.Parameter(torch.randn(self.num_codebooks, self.embedding_dim))
        # Quantization dropout
        self.quantization_dropout = nn.Dropout(config.quantization_dropout) if config.quantization_dropout > 0 else None

    def forward(self, x):
        # Flatten the input for calculating distances
        flat_input = x.view(-1, self.embedding_dim)

        # Compute distances to each codebook embedding
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self.codebook.t()))

        # Get the closest codebook entries
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = torch.index_select(self.codebook, dim=0, index=encoding_indices.squeeze()).view_as(x)

        # Apply dropout to quantized values
        if self.quantization_dropout:
            quantized = self.quantization_dropout(quantized)

        # Compute commitment loss
        commitment_loss = self.commitment_cost * torch.mean((quantized.detach() - x) ** 2)
        quantized = x + (quantized - x).detach()  # Straight-through estimator for backpropagation

        return quantized, commitment_loss