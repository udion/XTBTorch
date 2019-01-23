import torch
import torchvision
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from gen_utils import *

class XrayDset(Dataset):
    def __init__(self, root_dir, tfm=transforms.ToTensor()):
        self.root_dir = root_dir
        self.fnames = os.listdir(root_dir)
        self.labels = [get_lbl_from_name(f) for f in self.fnames]
        self.tfm = tfm
        
    def __getitem__(self, index):
        I0 = Image.open(self.root_dir+self.fnames[index])
        I = self.tfm(I0)
        return I, self.labels[index]
    
    def __len__(self):
        return len(self.fnames)