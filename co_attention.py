import torch.nn as nn
from transformer.SubLayers import PositionwiseFeedForward


class CoTransformerEncoder(nn.Module):
    '''Co-Attention Transformer encoder. Inspired by ViLBERT model'''

    def __init__(self, q_dim, v_dim, hidden_dim=2048, num_heads = 12, drop_out=0.1) -> None:
        super(CoTransformerEncoder, self).__init__()
        # Transformer Encoder 1 for Visual features
        self.mhsa_1 = nn.MultiheadAttention(v_dim, kdim=q_dim, vdim=q_dim, num_heads=num_heads, batch_first=True, dropout=drop_out)
        self.ffw_1 = PositionwiseFeedForward(v_dim, hidden_dim, drop_out)
        self.norm_1 = nn.LayerNorm(q_dim)
        self.dropout_1 = nn.Dropout(drop_out)
        
        # Transformer Encoder 2 for Question features
        self.mhsa_2 = nn.MultiheadAttention(q_dim, kdim=v_dim, vdim=v_dim, num_heads=num_heads, batch_first=True, dropout=drop_out)
        self.ffw_2 = PositionwiseFeedForward(q_dim, hidden_dim, drop_out)
        self.norm_2 = nn.LayerNorm(v_dim)
        self.dropout_2 = nn.Dropout(drop_out)
            
    def forward(self, v, q):
        ''' Forward
        v: [batch, k, v_dim]
        q: [batch, seq_len, q_dim]
        '''
        x_1, _ = self.mhsa_1(v, q, q)
        x_1 = self.dropout_1(self.norm_1(x_1) + v)
        x_1 = self.ffw_1(x_1)
        
        x_2, _ = self.mhsa_2(q, v, v)
        x_2 = self.dropout_2(self.norm_2(x_2) + q)
        x_2 = self.ffw_2(x_2)
        
        return x_1, x_2


class CoTransformerBlock(nn.Module):
    def __init__(self, q_dim, v_dim, num_head_trfm=12, hidden_size=2048, drop_out=0.1) -> None:
        super(CoTransformerBlock, self).__init__()
        self.co_trm_encoder = CoTransformerEncoder(q_dim, v_dim, hidden_size, num_head_trfm, drop_out)
        self.transformer_1 = nn.TransformerEncoderLayer(q_dim, nhead=num_head_trfm, dim_feedforward=hidden_size, dropout=drop_out, batch_first=True)
        self.transformer_2 = nn.TransformerEncoderLayer(q_dim, nhead=num_head_trfm, dim_feedforward=hidden_size, dropout=drop_out, batch_first=True)
        
    def forward(self, v, q):
        ''' Forward
        v: [batch, k, v_dim]
        q: [batch, seq_len, q_dim]
        '''
        v_out, q_out = self.co_trm_encoder(v, q)
        v_out = self.transformer_1(v_out)
        q_out = self.transformer_2(q_out)
        return v_out, q_out


class FusionAttentionFeature(nn.Module):
    def __init__(self, args) -> None:
        super(FusionAttentionFeature, self).__init__()
        self.q_convert = nn.Sequential(*[
            nn.Linear(args.q_dimn, args.f_mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.f_mid_dim, args.joint_dim)
        ])
        self.v_convert = nn.Sequential(*[
            nn.Linear(args.v_dimn, args.f_mid_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(args.f_mid_dim, args.joint_dim)
        ])
        self.layer_norm = nn.LayerNorm(args.joint_dim)

    def forward(self, v_feat, q_feat):
        '''Forward
        v_feat: [batch, v_len, v_dim]
        q_feat: [batch, q_len, q_dim]
        '''
        v_converted = self.v_convert(v_feat)
        q_converted = self.q_convert(q_feat)

        out = self.layer_norm(v_converted + q_converted)

        return out