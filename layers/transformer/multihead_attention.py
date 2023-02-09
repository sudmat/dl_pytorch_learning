from torch import nn
import torch
import numpy as np
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt

class MultiheadAttentionNotClever(nn.Module):

    def __init__(self, num_head, in_dim, out_dim, key_dim, value_dim):
        super().__init__()
        self.num_head = num_head
        self.LinearQ = [nn.Linear(in_features=in_dim, out_features=key_dim) for i in range(num_head)]
        self.LinearK = [nn.Linear(in_features=in_dim, out_features=key_dim) for i in range(num_head)]
        self.LinearV = [nn.Linear(in_features=in_dim, out_features=value_dim) for i in range(num_head)]
        self.LinearOut = nn.Linear(in_features=value_dim * num_head, out_features=out_dim)
        # here we set dim=-1 because we want the row-sum of the attention matrix to be 1. If dim=1, col-sum is 1.
        self.softmax = nn.Softmax(dim=-1)
    
    def scaled_dot_product_attention(self, Q, K, V):
        dk = Q.shape[-1]
        attn = self.softmax((Q @ K.transpose(-2, -1)) / np.sqrt(dk))
        newV = attn @ V

        # compare with the implementation in the tutorial
        v_gt, attn_gt = self.scaled_dot_product_attention_gt(Q, K, V)

        return newV, attn
    
    def scaled_dot_product_attention_gt(self, q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x):

        Qs = [L(x) for L in self.LinearQ]
        Ks = [L(x) for L in self.LinearK]
        Vs = [L(x) for L in self.LinearV]

        As = [self.scaled_dot_product_attention(Q, K, V)[0] for Q, K, V in zip(Qs, Ks, Vs)]
        A = torch.cat(As, axis=-1)

        y = self.LinearOut(A)

        return y
        
class MultiheadAttention(nn.Module):

    def __init__(self, in_dim, out_dim, num_head):
        super().__init__()
        self.num_head = num_head
        self.out_dim = out_dim
        self.hidden_dim = out_dim // num_head
        self.qkv_linear = nn.Linear(in_features=in_dim, out_features=3 * out_dim)
        self.out_linear = nn.Linear(in_features=out_dim, out_features=out_dim)
        
        self._reset_parameters()
    
    def same_as_gt(self, gt_layer):
        self.qkv_linear = gt_layer.qkv_proj
        self.out_linear = gt_layer.o_proj
    
    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        # t1 = self.qkv_linear.weight.data.cpu().numpy().copy()
        nn.init.xavier_uniform_(self.qkv_linear.weight)
        # t2 = self.qkv_linear.weight.data.cpu().numpy().copy()

        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(t1)
        # axs[1].imshow(t2)

        # plt.show()
        self.qkv_linear.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_linear.weight)
        self.out_linear.bias.data.fill_(0)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        dk = Q.shape[-1]
        attn_logits = (Q @ K.transpose(-2, -1)) / np.sqrt(dk)
        if mask is not None:
            # why not set to zero?
            attn_logits = attn_logits.masked_fill(mask == 0, 1e-15)
        attn = F.softmax(attn_logits, dim=-1)
        newV = attn @ V
        return newV, attn

    def forward(self, x, mask=None, return_attn=False):

        qkv = self.qkv_linear(x)
        qkv = qkv.reshape(qkv.shape[0], qkv.shape[1], self.num_head, 3 * self.hidden_dim)
        qkv = qkv.permute((0, 2, 1, 3))
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        new_v, attn = self.scaled_dot_product_attention(q, k, v, mask)
        new_v = new_v.permute((0, 2, 1, 3))
        new_v = torch.flatten(new_v, start_dim=2, end_dim=-1)

        out = self.out_linear(new_v)

        if return_attn:
            return out, attn

        return out

class MultiheadAttentionGT(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()
    
    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o