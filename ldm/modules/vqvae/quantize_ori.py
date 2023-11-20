import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from functools import partial
from typing import Tuple, Optional

class VectorQuantizer(nn.Module):
    def __init__(self, n_embed: int, embed_dim: int, beta: float = 0.25, straight_through: bool = True,
                 use_norm: bool = True, use_residual: bool = False, num_quantizers: Optional[int] = None,
                 legacy: bool = True, **kwargs):
        super().__init__()
        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta
        self.legacy = legacy

        self.straight_through = straight_through
        self.norm = lambda x: F.normalize(x, dim=-1) if use_norm else x

        self.use_residual = use_residual
        self.num_quantizers = num_quantizers


        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_embed, 1.0 / self.n_embed)

    def quantize(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z_reshaped_norm = self.norm(z.view(-1, self.embed_dim))
        embedding_norm = self.norm(self.embedding.weight)

        d = torch.sum(z_reshaped_norm ** 2, dim=1, keepdim=True) + \
            torch.sum(embedding_norm ** 2, dim=1) - 2 * \
            torch.einsum('b d, n d -> b n', z_reshaped_norm, embedding_norm)
        encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        encoding_indices = encoding_indices.view(*z.shape[:-1])
        z_q = self.embedding(encoding_indices).view(z.shape)
        z_qnorm, z_norm = z_q, z

        # compute loss for embedding
        loss = self.beta * torch.mean((z_qnorm.detach() - z_norm) ** 2) + \
               torch.mean((z_qnorm - z_norm.detach()) ** 2)

        return z_qnorm, loss, encoding_indices

    def forward(self, z: torch.FloatTensor) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.LongTensor]:
        z = rearrange(z, 'b c d h w -> b d h w c').contiguous()
        if not self.use_residual:
            z_q, loss, encoding_indices = self.quantize(z)
        else:
            z_q = torch.zeros_like(z)
            residual = z.detach().clone()

            losses = []
            encoding_indices = []

            for _ in range(self.num_quantizers):
                z_qi, loss, indices = self.quantize(residual.clone())
                residual.sub_(z_qi)
                z_q.add_(z_qi)

                encoding_indices.append(indices)
                losses.append(loss)

            losses, encoding_indices = map(partial(torch.stack, dim = -1), (losses, encoding_indices))
            loss = losses.mean()

        # preserve gradients with straight-through estimator
        if self.straight_through:
            z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b d h w c -> b c d h w').contiguous()
        return z_q, loss, encoding_indices


class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5):
        super().__init__()
        self.decay = decay
        self.eps = eps        
        weight = torch.randn(num_tokens, codebook_dim)
        self.weight = nn.Parameter(weight, requires_grad = False)
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)
        self.update = True

    def forward(self, embed_id):
        return F.embedding(embed_id, self.weight)

    def cluster_size_ema_update(self, new_cluster_size):
        self.cluster_size.data.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

    def embed_avg_ema_update(self, new_embed_avg): 
        self.embed_avg.data.mul_(self.decay).add_(new_embed_avg, alpha=1 - self.decay)

    def weight_update(self, num_tokens):
        n = self.cluster_size.sum()
        smoothed_cluster_size = (
                (self.cluster_size + self.eps) / (n + num_tokens * self.eps) * n
            )
        #normalize embedding average with smoothed cluster size
        embed_normalized = self.embed_avg / smoothed_cluster_size.unsqueeze(1)
        self.weight.data.copy_(embed_normalized)   


if __name__ == '__main__':
    data = torch.randn(1,24,32,32,32)
    v = VectorQuantizer(embed_dim=24, n_embed=1024)
