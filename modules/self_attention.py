# This script is for self attention of user history modeling
import torch
from torch import nn


class MultiheadSelfAttention(nn.Module):
    """
    Multi-headed self attention
    """
    def __init__(
            self,
            input_dim,
            embed_dim,  # q,k,v have the same dimension here
            num_heads=1,  # By default, we use single head
                 ):
        super().__init__()
        self.q_proj = nn.Linear(input_dim, embed_dim)
        self.k_proj = nn.Linear(input_dim, embed_dim)
        self.v_proj = nn.Linear(input_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
        )

    def forward(self, x):
        # x: (Time, Batch_Size, Channel)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output, _ = self.attention(q, k, v)
        return attn_output  # (Time, Batch_Size, embed_dim)
