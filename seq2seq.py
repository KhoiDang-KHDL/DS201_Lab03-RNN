import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        # Encoder: 5 layers LSTM
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        # src: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(src))
        
        # outputs: [batch, seq_len, hid_dim]
        # hidden/cell: [n_layers, batch, hid_dim]
        outputs, (hidden, cell) = self.lstm(embedded)
        
        # Trả về hidden và cell state để khởi tạo cho Decoder
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        # Embedding cho các nhãn (Tags)
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        # Decoder: 5 layers LSTM
        self.lstm = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        # input: [batch_size] (1 tag tại 1 thời điểm)
        # input phải được unsqueeze để thành [batch_size, 1]
        input = input.unsqueeze(1)
        
        embedded = self.dropout(self.embedding(input))
        
        # Decoder chạy từng bước một (seq_len = 1)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # output: [batch, 1, hid_dim] -> squeeze -> [batch, hid_dim]
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, "Hidden dimensions of Encoder and Decoder must match!"
        assert encoder.n_layers == decoder.n_layers, "Number of layers of Encoder and Decoder must match!"
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, seq_len]
        # trg: [batch, seq_len] (Chứa nhãn thực tế)
        
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        # Tensor để chứa kết quả dự đoán toàn bộ câu
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 1. Encode: Lấy context vector từ Encoder
        hidden, cell = self.encoder(src)
        
        # 2. Decode:
        # Input đầu tiên cho Decoder là token bắt đầu (thường là <SOS> hoặc tag đầu tiên của trg)
        # Ở đây ta dùng tag đầu tiên của trg (thường là padding hoặc tag đầu câu)
        input = trg[:, 0] 
        
        for t in range(1, trg_len):
            # Chạy Decoder 1 bước
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            # Lưu dự đoán
            outputs[:, t, :] = output
            
            # Quyết định dùng Teacher Forcing hay không
            # Teacher Forcing: Dùng nhãn thật (trg) làm input tiếp theo
            # Không dùng: Dùng dự đoán cao nhất của model làm input tiếp theo
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 
            
            input = trg[:, t] if teacher_force else top1
            
        return outputs