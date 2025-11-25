import torch
from torch import nn

class GRUModel(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int,
        hidden_size: int, 
        n_layers: int, 
        n_labels: int, 
        dropout: float = 0.2, 
        padding_idx: int = 0
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )
        
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layers,
            bidirectional=True, # Đọc 2 chiều
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Vì dùng 2 chiều nên đầu ra nhân đôi (hidden_size * 2)
        self.classifier = nn.Linear(
            in_features=hidden_size * 2, 
            out_features=n_labels
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor):
        # 1. Embedding
        embedded_feats = self.embedding(input_ids)
        
        # 2. GRU
        # output: chứa features của toàn bộ các bước thời gian
        # h_n: chứa trạng thái ẩn cuối cùng của các lớp 
        output, h_n = self.gru(embedded_feats)
        
        # 3. Lấy (Context Vector)
        # h_n có kích thước: [num_layers * num_directions, batch, hidden_size]
        
        # Lấy hidden state của lớp cuối cùng (last layer)
        # [-2] là chiều thuận (forward), [-1] là chiều nghịch (backward)
        feature_fwd = h_n[-2, :, :]
        feature_bwd = h_n[-1, :, :]
        
        # Ghép 2 vector
        context_vector = torch.cat((feature_fwd, feature_bwd), dim=1)
        
        # 4. Phân lớp
        context_vector = self.dropout(context_vector)
        logits = self.classifier(context_vector)
        
        return logits
