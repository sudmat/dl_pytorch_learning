from torch.utils.data import Dataset
import torch

class ReversedSequence(Dataset):

    def __init__(self, num_categories, seq_len, num_seqs) -> None:
        super().__init__()
        
        self.data = torch.randint(num_categories, size=(num_seqs, seq_len))
    
    def __getitem__(self, index):
        seq = self.data[index]
        label = seq.flip(dims=(0, ))
        return seq, label
    
    def __len__(self):
        return self.data.shape[0]