import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import os

# --- HÀM ĐỌC DỮ LIỆU THÔNG MINH ---
def load_data_smart(filepath):
    """Đọc được cả file JSON thường và JSON Lines"""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                data = content
            elif isinstance(content, dict):
                data = list(content.values())
    except json.JSONDecodeError:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except:
                        continue
    return data

def collate_fn(items: list) -> dict:
    input_ids = [item['input_ids'] for item in items]
    label_ids = [item['label'] for item in items]
    
    # QUAN TRỌNG: Padding value cho cả Input và Label đều là 0
    # Để tránh lỗi CUDA khi đưa vào Embedding layer
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_label_ids = pad_sequence(label_ids, batch_first=True, padding_value=0)
    
    return {
        'input_ids': padded_input_ids,
        'label': padded_label_ids
    }

class Vocab:
    def __init__(self, path: str):
        all_words = set()
        all_tags = set()
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Lỗi: Không tìm thấy thư mục {path}")

        print(f"--- Đang quét dữ liệu tại: {path} ---")
        files_found = False

        for filename in os.listdir(path):
            full_filepath = os.path.join(path, filename)
            
            if os.path.isfile(full_filepath) and filename.endswith('.json'):
                files_found = True
                data = load_data_smart(full_filepath)
                
                for item in data:
                    words = item.get('words', [])
                    tags = item.get('tags', [])
                    
                    for w in words:
                        all_words.add(w.lower())
                    for t in tags:
                        all_tags.add(t)
        
        if not files_found:
            print("CẢNH BÁO: Không tìm thấy file .json nào!")

        # 1. Xây dựng từ điển TỪ (Words)
        self.bos = "<s>"
        self.pad = "<p>"
        self.unk = "<unk>"
        
        self.w2i = {word: idx for idx, word in enumerate(all_words, start=3)}
        self.w2i[self.pad] = 0  # Pad ID = 0
        self.w2i[self.bos] = 1
        self.w2i[self.unk] = 2
        
        self.i2w = {idx: word for word, idx in self.w2i.items()}
        
        # 2. Xây dựng từ điển NHÃN (Tags) - SỬA LẠI ĐỂ TRÁNH LỖI CUDA
        # Tag <pad> bắt buộc phải là 0 để khớp với padding_value trong collate_fn
        self.tag_pad = "<pad>"
        self.tag2i = {self.tag_pad: 0}
        
        sorted_tags = sorted(list(all_tags))
        for idx, tag in enumerate(sorted_tags, start=1):
            self.tag2i[tag] = idx
            
        self.i2tag = {idx: tag for tag, idx in self.tag2i.items()}
        
        print(f"-> Đã xây dựng Vocab: {len(self.w2i)} từ, {len(self.tag2i)} nhãn (bao gồm pad).")
        
    @property 
    def n_labels(self) -> int:
        return len(self.tag2i)
    
    @property
    def len(self) -> int:
        return len(self.w2i)
    
    def encode_words(self, words: list) -> torch.Tensor:
        words_ids = []
        for w in words:
            w_lower = w.lower()
            words_ids.append(self.w2i.get(w_lower, self.w2i[self.unk]))
        return torch.tensor(words_ids).long()
    
    def encode_tags(self, tags: list) -> torch.Tensor:
        tag_ids = [self.tag2i[t] for t in tags]
        return torch.tensor(tag_ids).long()

class PhoNER(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()
        self.vocab = vocab
        self._data = []
        
        if os.path.exists(path):
            self._data = load_data_smart(path)
            print(f"-> Đã tải {len(self._data)} mẫu từ {os.path.basename(path)}")
        else:
            raise FileNotFoundError(f"Không tìm thấy file: {path}")

        if len(self._data) == 0:
             raise ValueError(f"File {path} rỗng hoặc sai định dạng!")

    def __len__(self) -> int:
        return len(self._data)
    
    def __getitem__(self, index: int) -> dict:
        item = self._data[index]
        words = item.get('words', [])
        tags = item.get('tags', [])
        
        input_ids = self.vocab.encode_words(words)
        label_ids = self.vocab.encode_tags(tags)
        
        # Cắt nếu độ dài không khớp (phòng hờ dữ liệu lỗi)
        min_len = min(len(input_ids), len(label_ids))
        return {
            'input_ids': input_ids[:min_len],
            'label': label_ids[:min_len]
        }
