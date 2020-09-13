import json
from PIL import Image
import os
from torch.utils.data import Dataset,DataLoader

class LandslideDataSet(Dataset):
    def __init__(self,root,transforms=None,train_set=True):
        self.root=os.path.join(root,"data1")
        self.img_root=os.path.join(self.root,"image")

        if train_set:
            txt=os.path.join(self.root,"main","train.txt")
        else:
            txt=os.path.join(self.root,"main","val.txt")





