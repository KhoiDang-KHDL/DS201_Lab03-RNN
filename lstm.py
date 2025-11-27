import torch
from torch import nn

class LSTMModel(nn.Module):
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
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=n_layers,
            bidirectional=True, 
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Vì dùng bidirectional, hidden_size đầu ra sẽ gấp đôi
        self.classifier = nn.Linear(
            in_features=hidden_size * 2, 
            out_features=n_labels
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor):
        # input_ids: [Batch, Seq_Len]
        embedded = self.embedding(input_ids)
        
        # output: [Batch, Seq_Len, Hidden*2]
        # h_n: [Num_Layers * 2, Batch, Hidden]
        # c_n: [Num_Layers * 2, Batch, Hidden]
        output, (h_n, c_n) = self.lstm(embedded)
        
        # Lấy hidden state của layer cuối cùng.
        # h_n chứa trạng thái cuối của cả chiều thuận và chiều nghịch
        # Cấu trúc h_n: [layer_1_fwd, layer_1_bwd, ..., layer_n_fwd, layer_n_bwd]
        
        # Lấy layer cuối cùng, chiều thuận và chiều nghịch ghép lại
        feature_fwd = h_n[-2, :, :]
        feature_bwd = h_n[-1, :, :]
        
        # Ghép lại: [Batch, Hidden * 2]
        features = torch.cat((feature_fwd, feature_bwd), dim=1)
        
        features = self.dropout(features)
        logits = self.classifier(features)
        
        return logits
