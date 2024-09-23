
import torch 
from torch import nn
import torch.nn.functional as F
from local_attention import LocalAttention
from pianogen.pe import binary_positional_encoding, sinusoidal_positional_encoding

class BinaryPositionalEncoding(nn.Module):
    '''
    Input: B, L (long)
    Output: B, L, D
    '''
    def __init__(self, dim:int, max_len:int):
        super().__init__()
        self.register_buffer('pos_encoding', binary_positional_encoding(max_len, dim).unsqueeze(0))

    def forward(self, pos: torch.Tensor):
        return torch.gather(self.pos_encoding.expand(pos.shape[0], -1, -1), 1, pos.unsqueeze(-1).expand(-1, -1, self.pos_encoding.shape[-1]))
    
class SinusoidalPositionalEncoding(nn.Module):
    '''
    Input: B, L (long)
    Output: B, L, D
    '''
    def __init__(self, dim:int, max_len:int):
        super().__init__()
        self.register_buffer('pos_encoding', sinusoidal_positional_encoding(max_len, dim).unsqueeze(0))

    def forward(self, pos: torch.Tensor):
        return torch.gather(self.pos_encoding.expand(pos.shape[0], -1, -1), 1, pos.unsqueeze(-1).expand(-1, -1, self.pos_encoding.shape[-1]))
    
class LocalMultiHeadAttention(nn.Module):
    '''
    Input: B, L, D
    Output: B, L, D
    '''
    def __init__(self, heads, dim, window_size, causal = False, dropout = 0.):
        super().__init__()
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        self.heads = heads
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.local_attn = LocalAttention(dim = dim // heads, window_size = window_size, causal = causal, dropout = dropout, autopad=True)

    def forward(self, x, mask = None):
        B, L, D = x.shape
        H = self.heads
        E = D // H

        qkv = self.to_qkv(x).chunk(3, dim = -1) # B, L, 3 * H, E
        q, k, v = map(lambda t: t.view(B, L, H, E).transpose(1, 2), qkv)

        out = self.local_attn(q, k, v, mask = mask)
        out = out.transpose(1, 2).reshape(B, L, D)
        return out

class LMHATransformerBlock(nn.Module):
    '''
    Input: B, L, D
    Output: B, L, D
    '''
    def __init__(self, dim, heads, window_size, dim_feedforward, dropout = 0., causal = False):
        super().__init__()
        self.attn = LocalMultiHeadAttention(heads = heads, dim = dim, window_size = window_size, dropout = dropout, causal = causal)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim_feedforward),
            nn.LeakyReLU(),
            nn.Linear(dim_feedforward, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask = None):
        x = x + self.dropout(self.attn(self.norm1(x), mask = mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

class SelectiveAttnTransformer(nn.Module):
    '''
    Token level attention is too expensive to apply on the whole sequence. This module instead learns a regular attention mask with
    a downsampled sequence (segment level), then transform it into the mask for the token level attention, by sparsely select the
    most important segments (the selection is not differentiable though).

    As such, token level attention is only applied on the selected segments, which is much faster.

    Input: B, n_token, n_feature
    '''

    def __init__(self, vocab_size, segment_len, dim = 256):
        super().__init__()

        self.segment_len = segment_len
        self._downsampled_path_enabled = True

        self.binary_pe_dim = 5
        self.sinusoidal_pe_dim = 123
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.binary_pos_encoding = BinaryPositionalEncoding(self.binary_pe_dim, 10240)
        self.sinusoidal_pos_encoding = SinusoidalPositionalEncoding(self.sinusoidal_pe_dim, 10240)

        self.in_local_attention = nn.Sequential(*[
            LMHATransformerBlock(heads=8, dim=dim, window_size=200, causal=True, dropout=0.1, dim_feedforward=1024) for _ in range(2)
        ])
        self.downsample = nn.AvgPool1d(segment_len, stride=segment_len)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=8, dim_feedforward=1024, batch_first=True), num_layers=4)
        self.upsample = nn.Upsample(scale_factor=segment_len, mode='nearest')
        self.out_local_attention = nn.Sequential(*[
            LMHATransformerBlock(heads=8, dim=dim, window_size=200, causal=True, dropout=0.1, dim_feedforward=1024) for _ in range(2)
        ])
        self.out_linear = nn.Linear(dim, vocab_size)

    def forward(self, x, pos):
        # x: B, L
        # pos: B, L+1
        x = self.token_embedding(x)
        x = self.forward_after_token_embedding(x, pos)
        return x
    
    def get_jacobian(self, x, pos, target_range: range):
        '''
        Used to inspect relevance of input features to the output.

        x: int tensor of shape (B, L) where B is 1 (only the first sample in the batch is used if B > 1)

        pos: int tensor of shape (B, L+1) where B is 1 (only the first sample in the batch is used if B > 1)

        target_range: range of output indices to calculate jacobian for. Do not make it too large as it will consume too much memory.
        '''
        x = self.token_embedding(x)
        jacobian = torch.autograd.functional.jacobian(lambda x: self.forward_after_token_embedding(x, pos)[0,target_range].abs().sum(1,keepdim=False), x)
        # dim  1 is batch dim, squeeze it
        jacobian = jacobian.squeeze(1)
        return jacobian
    
    def test_causal(self):
        # jacobian have to be lower triangular. If not, there is information leak from future tokens to past tokens
        size = self.segment_len * 4
        x = torch.randint(0, self.token_embedding.num_embeddings, (1, size)).to(self.token_embedding.weight.device)
        pos = torch.arange(size+1).unsqueeze(0).to(self.token_embedding.weight.device)
        jacobian = self.get_jacobian(x, pos, range(size)).norm(dim=2)
        upper_triangular = torch.triu(torch.ones(size, size), diagonal=1).to(jacobian.device)
        assert torch.allclose(jacobian * upper_triangular, torch.zeros_like(jacobian)), 'jacobian is not lower triangular'
        print('test_causal passed')

    
    def forward_after_token_embedding(self, x, pos):
        # x: B, L, D
        # pos: B, L+1

        B, L, _ = x.shape
        pe = torch.cat([
            self.binary_pos_encoding(pos),
            self.sinusoidal_pos_encoding(pos),
        ], dim=-1) # B, L+1, D/2

        pe = torch.cat([
            pe[:, :-1], # pe of the input tokens
            pe[:, 1:]   # pe of the target tokens
        ], dim=2) # B, L, D

        x = x + pe
        
        
        x = self.in_local_attention(x)
        before_down = x

        # before entering the downsampled path, we need to make L % segment_len == 0

        if L >= self.segment_len and self._downsampled_path_enabled: # if L < segment_len, we don't need to downsample

            x = x[:, :L - (L % self.segment_len)] # B, L - (L % segment_len), D

            x = self.downsample(x.transpose(1, 2)).transpose(1, 2)

            x = self.transformer(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device), is_causal = True)
            x = self.upsample(x.transpose(1, 2)).transpose(1, 2) # B, L - (L % segment_len), D

            # to avoid information leak, shift the data from the downsampled path right by segment_len-1
            x = F.pad(x, (0,0,self.segment_len-1, 0), 'constant', 0) # B, L+self.segment_len-1-(L % self.segment_len), D

            # crop redundant right-most stuff due to the right shift
            x = x[:, :L] # B, L, D
            # skip connection
            x = x + before_down


        x = self.out_local_attention(x)
        x = F.leaky_relu(x)
        x = self.out_linear(x)
        return x
    
    def set_downsampled_path_enabled(self, enabled):
        self._downsampled_path_enabled = enabled

class PianoRollGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_linear = nn.Linear(200, 256)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, batch_first=True), num_layers=6)
        self.out_linear = nn.Linear(256, 121)

    def forward(self, x):
        x = self.in_linear(x)
        x = self.transformer(x, mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1]).to(x.device), is_causal = True)
        x = self.out_linear(x)
        return x
        
