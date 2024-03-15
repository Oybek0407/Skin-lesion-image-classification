import timm, os, torch, numpy as np
from tqdm import tqdm
from data import get_dl
def set_up(model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cuda")
    model = model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=3e-4)
    criterion = torch.nn.CrossEntropyLoss()
    epochs = 10
    return device, model, optimizer, criterion, epochs
        
def train(model, tr_dl, val_dl, loss_fn, epochs, opt, device, save_prefix, save_dir, threshold):
    tr_acc_sc, tr_loss_cs, val_acc_sc, val_loss_cs =[],[],[],[]
    best_loss = np.inf
    for epoch in range(epochs):
        print(f"{epoch+1} - epoch is starting ....")
        tr_loss, tr_acc = 0,0
        for idx, batch in enumerate(tqdm(tr_dl)):
           
            im, gt = batch
            im, gt = im.to(device), gt.to(device)
            pred = model(im)
            loss = loss_fn(pred, gt)
            tr_loss+=loss.item()
            
            pred_class = torch.argmax(pred, dim =1)
            tr_acc+=(pred_class==gt).sum().item()
            # preform optimizetion steps
            opt.zero_grad(); loss.backward(); opt.step()

        tr_loss /= len(tr_dl)
        tr_acc /= len(tr_dl.dataset)
        tr_acc_sc.append(tr_acc); tr_loss_cs.append(tr_loss)
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = 0,0
            for idx, batch in enumerate(tqdm(val_dl)):
                im, gt = batch
                im, gt = im.to(device), gt.to(device)
                pred = model(im)
                loss = loss_fn(pred, gt)
               
                val_loss +=loss.item()
                pred_class = torch.argmax(pred, dim=1)
                val_acc += (pred_class==gt).sum().item()
                
            val_acc /= len(val_dl.dataset)
            val_loss /= len(val_dl)
            val_acc_sc.append(val_acc); val_loss_cs.append(val_loss)
            
            print(f"{epoch+1} - epoc Train process is results:\n")
            print(f"{epoch+1} - epoc Train Accuracy score       - > {tr_acc:.3f}")
            print(f"{epoch+1} - epoc Train epoc loss  score      - > {tr_loss:.3f}")
            print(f"{epoch+1} - epoc Validation process is results:\n")
            print(f"{epoch+1} - epoc Validation Accuracy score  - > {val_acc:.3f}")
            print(f"{epoch+1} - epoc Validation epoc loss  score - > {val_loss:.3f}")
            
            if val_loss < (best_loss + threshold):
                best_loss = val_loss
                os.makedirs(save_dir, exist_ok= True)
                torch.save(model.state_dict(), f"{save_dir}/{save_prefix}_best_model.pth")
    return {"tr_acc_sc": tr_acc_sc, "tr_loss_cs": tr_loss_cs, "val_acc_sc": val_acc_sc, "val_loss_cs": val_loss_cs}