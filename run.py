import argparse, timm, torch, random, pandas as pd, numpy as np,  pickle as p
from matplotlib import pyplot as plt
from torchvision import transforms as T
from data import get_dl
from utils import Visualize, Learning_curve
from train import train
from train import set_up
from inference import inference
from tqdm import tqdm

def run(args):
  
        mean, std, im_size=[0.2250, 0.2250, 0.2250], [0.2505, 0.2505, 0.2505],224
        tfs = T.Compose([T.Resize((im_size, im_size)), T.ToTensor(), T.Normalize(mean = mean, std = std)])
        tr_dl, val_dl, ts_dl , classes = get_dl(root =args.data_path, transforamtions= tfs, bs = args.batch_size)
        print("1 -> Everything is okay ............")
        data_name = {tr_dl : "train", val_dl: "validation", ts_dl: "test"}
        print("\n2 -> Sample datas are saving .........")
        for data, name in data_name.items():
                Visualize(data = data.dataset, rows = args.rows, num_ims = args.num_im, cmap = None, class_name = list(classes.keys()), save_file = args.save_file , name = name)

        print(f"\n Sample images are being saved in a file named {args.save_file}!\n")
        model = timm.create_model(model_name = args.model_name, pretrained = True, num_classes = len(classes))
        set_up(model = model)
        device, model, optimizer, criterion, epochs = set_up(model)
        result = train(model = model, tr_dl = tr_dl, val_dl = val_dl, loss_fn = criterion, epochs = epochs, opt = optimizer,
                       device = device, save_prefix = args.save_prefix, save_dir = args.save_dir, threshold = 0.001)
        print(f"Train processing is finneshed best model is saved in the {args.save_prefix} file")
        # learning Curve
        Learning_curve(result = result, save_file = args.learn_curve, name = args.file_name)
        print(f"learning curve results are saved in the {args.learn_curve} file")

       
        
        # inference process
        model.load_state_dict(torch.load(f"{args.save_load}/{args.save_fol}_best_model.pth"))
        model.eval()
        inference(model = model, data = ts_dl, device = device, num_im = args.num_image, row = 4, class_name = list(classes.keys()),
                  im_dim = args.im_dim, save_prefix=args.save_inf, name = args.save_files)

        print("All fineshed  수고 했에용")


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Skin Cancer  Classification Demo")
    parser.add_argument("-dp", "--data_path", type=str, default="data/skin_lesion"  , help="path of dataset")
    parser.add_argument("-bs", "--batch_size", type=str, default= 32  , help="path of dataset")
    # Visualize
    parser.add_argument("-rw", "--rows", type=str, default= 4  , help="rows of visualization images")
    parser.add_argument("-nm", "--num_im", type=str, default= 20  , help="  numbers images")
    parser.add_argument("-sf", "--save_file", type=str, default= "Skin cancer" , help="save file of sample images")
    # Model

    parser.add_argument("-mp", "--model_name", type=str, default= "rexnet_150" , help="AI train model")
        
    # train
    parser.add_argument("-sp", "--save_prefix", type=str, default= "Best_model" , help="file for saving best model")
    parser.add_argument("-sg", "--save_dir", type=str, default= "skin_best" , help="file-direct for saving best model")
    # Learning Curve
    parser.add_argument("-lc", "--learn_curve", type=str, default= "learning_Curve" , help="file for saving Learning curve")
    parser.add_argument("-na", "--file_name", type=str, default= "Skin_" , help="file-direct for saving best model")
    
   #  inference

    parser.add_argument("-sl", "--save_load", type=str, default= "skin_best" , help="file-direct for saving best model")
    parser.add_argument("-so", "--save_fol", type=str, default= "Best_model"  , help="file-direct for saving best model")
    parser.add_argument("-mi", "--num_image", type=str, default= 20 , help="numbe images")
    parser.add_argument("-im", "--im_dim", type=str, default= 224 , help="image dimention")
    parser.add_argument("-si", "--save_inf", type=str, default= "Inference_file" , help="file for saving Inference result")
    parser.add_argument("-sc", "--save_files", type=str, default= "Inference" , help="file for saving Inference result")
        
        
        
        
        
        
 
    args = parser.parse_args()
    run(args)
        
        

