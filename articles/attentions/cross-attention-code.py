import torch

class CrossAttention(torch.nn.Module):
    """
    Cross-Attention layer for processing two sequences of embeddings.
    This layer computes the attention scores between two sequences and generates
    a context vector for each element in the first sequence based on the second sequence.
    """
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.k_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.v_projection = torch.nn.Linear(embed_dim, embed_dim)
        self.out_projection = torch.nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key, value, mask = None):
        """
        Forward pass for the Cross-Attention layer.
        We first have basic dimension to obtains the batch(B), the size of the query(T_q) and the size key/value(T_k).
        Then we project the query, key and value using the linear layers defined in __init__.
        We reshape the projected query, key and value to separate the heads and then transpose them.
        We compute the attention scores by taking the dot product of the query and key, scaling it by the square root of the head dimension.
        If a mask is provided, we apply it to the scores.
        The mask is used to prevent attending to certain positions in the key/value sequences.
        We then apply softmax to the scores to obtain the attention weights.
        Finally, we compute the context vector by multiplying the attention weights with the value sequence.
        Args:
            query (torch.Tensor): The query sequence of shape (B, T_q, embed_dim).
            key (torch.Tensor): The key sequence of shape (B, T_k, embed_dim).
            value (torch.Tensor): The value sequence of shape (B, T_k, embed_dim).
            mask (torch.Tensor, optional): A mask of shape (B, 1, T_q, T_k) to prevent attending to certain positions.
        Returns:
            output (torch.Tensor): The output sequence of shape (B, T_q, embed_dim).
            attn_weights (torch.Tensor): The attention weights of shape (B, num_heads, T_q, T_k).
        """
        B, T_q, _ = query.size()
        T_k = key.size(1)
        Q = self.q_projection(query).view(B, T_q, self.num_heads, self.head_dim).transpose(1,2)
        K = self.k_projection(key).view(B, T_k, self.num_heads, self.head_dim).transpose(1,2)
        V = self.v_projection(value).view(B, T_k, self.num_heads, self.head_dim).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.embed_dim)
        output = self.out_projection(context)
        return output, attn_weights
