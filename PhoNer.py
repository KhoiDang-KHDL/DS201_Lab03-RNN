import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import json
import os
import string

def collate_fn(items: list) -> dict:
    # items là list các dict trả về từ __getitem__
    
    # Lấy list các input_ids và label_ids
    input_ids = [item['input_ids'] for item in items]
    label_ids = [item['label'] for item in items]
    
    # Padding cho Input (dùng số 0)
    # pad_sequence là hàm tiện lợi của PyTorch, thay vì viết tay vòng lặp
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    # Padding cho Label (QUAN TRỌNG: dùng -100)
    # Trong PyTorch CrossEntropyLoss, giá trị -100 mặc định sẽ bị bỏ qua không tính lỗi
    # Điều này giúp model không bị "học sai" ở những chỗ padding vô nghĩa
    padded_label_ids = pad_sequence(label_ids, batch_first=True, padding_value=-100)
    
    return {
        'input_ids': padded_input_ids,
        'label': padded_label_ids
    }

# --- 2. CLASS VOCAB ---
class Vocab:
    def __init__(self, path: str):
        all_words = set()
        all_tags = set()
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Không tìm thấy đường dẫn: {path}")

        # PhoNER thường là file .json (ví dụ: train_word.json)
        for filename in os.listdir(path):
            full_filepath = os.path.join(path, filename)
            
            if os.path.isfile(full_filepath) and filename.endswith('.json'):
                try:
                    with open(full_filepath, 'r', encoding='utf-8') as f:
                        # --- ĐOẠN SỬA BẮT ĐẦU ---
                        # Thay vì json.load(f), ta đọc từng dòng
                        for line in f:
                            line = line.strip()
                            if not line: continue # Bỏ qua dòng trống
                            
                            item = json.loads(line) # Đọc json từng dòng
                            
                            # Xử lý item
                            words = item.get('words', [])
                            tags = item.get('tags', [])
                            
                            for w in words:
                                all_words.add(w.lower())
                            
                            for t in tags:
                                all_tags.add(t)
                        # --- ĐOẠN SỬA KẾT THÚC ---
                            
                except Exception as e:
                    print(f"Lỗi khi đọc file {filename}: {e}")
                    continue
        
        self.bos = "<s>"
        self.pad = "<p>"
        self.unk = "<unk>" # Thêm token Unknown cho từ lạ
        
        # Xây dựng từ điển từ
        self.w2i = {word: idx for idx, word in enumerate(all_words, start=3)}
        self.w2i[self.pad] = 0
        self.w2i[self.bos] = 1
        self.w2i[self.unk] = 2
        
        self.i2w = {idx: word for word, idx in self.w2i.items()}
        
        # Xây dựng từ điển nhãn (Tags: B-LOC, I-PER, O, ...)
        self.tag2i = {tag: idx for idx, tag in enumerate(all_tags)}
        self.i2tag = {idx: tag for tag, idx in self.tag2i.items()}
        
    @property 
    def n_labels(self) -> int:
        return len(self.tag2i)
    
    @property
    def len(self) -> int:
        return len(self.w2i)
    
    def encode_words(self, words: list) -> torch.Tensor:
        # Input là List các từ (đã tách sẵn), không phải 1 câu string
        words_ids = []
        for w in words:
            w_lower = w.lower()
            # Nếu từ có trong từ điển thì lấy ID, không thì lấy ID của <unk>
            words_ids.append(self.w2i.get(w_lower, self.w2i[self.unk]))
            
        return torch.tensor(words_ids).long()
    
    def encode_tags(self, tags: list) -> torch.Tensor:
        tag_ids = [self.tag2i[t] for t in tags]
        return torch.tensor(tag_ids).long()
    
    def decode_tags(self, tag_ids: torch.Tensor) -> list:
        tag_ids = tag_ids.tolist()
        return [self.i2tag.get(idx, 'O') for idx in tag_ids]

# --- 3. CLASS DATASET ---
class PhoNER(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self._data = []
        
        if os.path.exists(path):
             with open(path, 'r', encoding='utf-8') as f:
                # --- ĐOẠN SỬA BẮT ĐẦU ---
                # Đọc file JSONL (từng dòng)
                for line in f:
                    line = line.strip()
                    if line:
                        self._data.append(json.loads(line))
                # --- ĐOẠN SỬA KẾT THÚC ---
        else:
            print(f"Cảnh báo: Không tìm thấy file {path}")

    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int) -> dict:
        item = self._data[index]
        
        # Dữ liệu PhoNER đã tách từ sẵn trong list "words"
        words = item['words'] 
        tags = item['tags']
        
        input_ids = self.vocab.encode_words(words)
        label_ids = self.vocab.encode_tags(tags)
        
        # Kiểm tra an toàn: độ dài từ và nhãn phải bằng nhau
        assert len(input_ids) == len(label_ids), f"Lỗi dữ liệu tại index {index}: độ dài từ và nhãn không khớp."
        
        return {
            'input_ids': input_ids,
            'label': label_ids
        }