import torch, gdown, os ,  cv2, random, pandas as pd, numpy as np, pickle as p
from matplotlib import pyplot as plt
from torchvision import transforms as T
def tr_2_im(t, type = "rgb"):
    gray = T.Compose([T.Normalize(mean = [ 0.], std = [1/0.5]), T.Normalize(mean = [-0.5], std = [1])])
    rgb = T.Compose([T.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.2505, 1/0.2505, 1/0.2505]),
                         T.Normalize(mean = [ -0.2250, -0.2250, -0.2250 ], std = [ 1., 1., 1. ])])
    inp = gray if type == "gray" else rgb
    return(inp(t)*255).detach().squeeze().cpu().permute(1,2,0).numpy().astype(np.uint8) if type == "gray" else (inp(t)*255).detach().cpu().permute(1,2,0).numpy().astype(np.uint8)


def Visualize(data, rows, num_ims, cmap = None, class_name =None, save_file = None, name = None):
    os.makedirs(save_file, exist_ok = True)
    if cmap == 'rgb': cmap = "RdBu"
    index = [random.randint(0, len(data)-1) for _ in range(num_ims)]
    plt.figure(figsize=(20, 15))
    for i, idx in enumerate(index):
        im, gt = data[idx]
        plt.subplot(rows, num_ims//rows, i+1)
        plt.imshow(tr_2_im(im, cmap), cmap = "RdBu")
        plt.imshow(tr_2_im(im))
        plt.title(f" gt - > {class_name[gt]}")
        plt.savefig(f"{save_file}/{name}_.png")

def Learning_curve(result, save_file =None, name = None):
        os.makedirs(save_file, exist_ok = True)
        plt.figure(figsize=(16, 10))
        plt.plot(result["tr_acc_sc"], label = "Train accurecy score")
        plt.plot(result["val_acc_sc"], label = "Validation accurecy score")
        plt.title("Train and validation Accuracy Score")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy Score")
        plt.xticks(np.arange(len(result["val_acc_sc"])), [result for result in range(1, len(result["val_acc_sc"])+1)])
        plt.legend()
        plt.savefig(f"{save_file}/{name}_Accurancy.png")

        plt.figure(figsize=(16, 10))
        plt.plot(result["tr_loss_cs"], label = "Train Loss score")
        plt.plot(result["val_loss_cs"], label = "Validation Loss score")
        plt.title("Train and validation Loss Score")
        plt.xlabel("Epochs")
        plt.ylabel("Loss Score")
        plt.xticks(np.arange(len(result["val_loss_cs"])), [result for result in range(1, len(result["val_loss_cs"])+1)])
        plt.legend()
        plt.savefig(f"{save_file}/{name}_Loss.png")
        

