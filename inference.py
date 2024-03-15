from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from tqdm import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as T
import timm, os, torch, numpy as np
from PIL import Image
from utils import tr_2_im
import torch, os, cv2, random

def inference(model, data, device, num_im, row, class_name = None, im_dim = None, save_prefix=None, name = None):
    os.makedirs(save_prefix, exist_ok  = True)
    Acc = 0
    preds, ims, gts = [],[],[]
    for idx, batch in enumerate(tqdm(data)):
        im,gt = batch
        im, gt = im.to(device), gt.to(device)
        pred = model(im)
    
        pred_class = torch.argmax(pred, dim = 1)
        Acc += (pred_class == gt).sum().item()
        ims.append(im); preds.append(pred_class.item()); gts.append(gt.item())
    print(f"Accuracy of the model on the test dataset -> {Acc/len(data): .3f}")
    plt.figure(figsize=(20,10))
    index = [random.randint(0, len(ims)-1) for _ in range(num_im)]
    
    for i, idx in enumerate(index):
        im = ims[idx].squeeze(); gt = gts[idx]; pred = preds[idx]
  
        #GradCAM
        orginal_im = tr_2_im(im)/255
        cam = GradCAMPlusPlus(model = model, target_layers=[model.features[-1]])
        grayscale_cam = cam(input_tensor = im.unsqueeze(0))[0, :]
        heat_map = show_cam_on_image(img= orginal_im, mask = grayscale_cam, image_weight=0.1,  use_rgb="jet")
        
        #start plot
        plt.subplot(row, num_im//row, i+1)
        plt.imshow(tr_2_im(im), cmap = "gray")
        plt.axis("off")
        plt.imshow(cv2.resize(heat_map, (im_dim, im_dim), interpolation = cv2.INTER_LINEAR ), alpha= 0.3, cmap = 'jet')
        plt.axis("off")
        color = ("green" if {class_name[int(gt)]} == {class_name[int(pred_class)]} else 'red')
        if class_name:
            plt.title(f"GT -> {class_name[gt]}; Pred- > {class_name[pred_class]}", color = color)
        else:
            plt.title(f"GT -> {gt}; PRED -> {pred}")
        plt.savefig(f"{save_prefix}/{name}.png")
        








