import pandas as pd, numpy as np, torch, cv2, os, random
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib import pyplot as plt
from PIL import Image
import argparse, timm, torch, random, pandas as pd, numpy as np,  pickle as p
from torchvision import transforms as T
class CustomDataset(Dataset):
    def __init__(self, root, transforamtions = None):
        super(). __init__()
        self.transforamtions = transforamtions
        self.ims = [] ; self.labels = []; self.class_name = {}; count = 0
        ds = pd.read_csv(f"{root}/meta_deta.csv")
        pixel = ds.loc[:, ds.columns != 'label'] # 
        label = ds["label"]
        for idx, pixel_label in enumerate(zip(pixel.values, label)):
            
            im, gt = pixel_label
            im_new_shape = np.reshape(im, newshape = (28,28,3))
            self.ims.append(im_new_shape)
            self.labels.append(int(gt))
            if gt not in self.class_name: self.class_name[int(gt)] = count; count += 1
    def __len__(self): return len(self.ims)

    def __getitem__(self, idx):
        im = Image.fromarray(self.ims[idx].astype("uint8"), "RGB")
        
        gt = self.class_name[self.labels[idx]]
        if self.transforamtions : im = self.transforamtions(im)
        return im, gt
       


def get_dl(root, transforamtions, bs, split = [0.9, 0.06, 0.04]):
    data = CustomDataset(root =root, transforamtions= transforamtions)
    len_data = len(data)
    tr_len = int(len_data * split[0])
    val_len = int(len_data * split[1])
    ts_len = len_data - (tr_len + val_len)
    tr_ds, val_ds, ts_ds = random_split(data, lengths= [tr_len, val_len , ts_len])
    
    tr_dl = DataLoader(dataset=tr_ds, batch_size=bs, shuffle=True, num_workers=0)
    val_dl = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False, num_workers=0)
    ts_dl = DataLoader(dataset=ts_ds, batch_size=1, shuffle=False, num_workers=0)
        
    save_prefix = "Skin"
    with open(f"{save_prefix}_classes_names.pickle", "wb") as f: p.dump(data.class_name, f, protocol = p.HIGHEST_PROTOCOL)
        
    return tr_dl, val_dl, ts_dl, data.class_name



