from numpy import cumproduct
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import torch
 
class RecoData(Dataset):
    def __init__(self, cfg, bfile, ufile):
        self.file = torch.LongTensor(bfile)
        self.cfg = cfg
        self.user_emb = torch.LongTensor(ufile)

    def __getitem__(self, index):
        curu = self.user_emb[self.file[index, -1]]
        return torch.cat([self.file[index, :-1], curu])
 
    def __len__(self):
        return self.file.size(0)

