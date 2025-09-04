import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_dim=512, embed_dim=512, num_heads=8, num_layers=4):
        super(TransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)    # [B, T, D] → [B, T, embed_dim]
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(embed_dim, input_dim)    # [B, T, embed_dim] → [B, T, D]

    def forward(self, x):
        x = self.input_proj(x)        # [B, T, D] → [B, T, embed_dim]
        x = self.transformer(x)       # [B, T, embed_dim]
        x = self.output_proj(x)       # [B, T, D]
        return x
