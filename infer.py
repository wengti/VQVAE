
import torch
from einops import rearrange
import torchvision
from pathlib import Path
import pickle

def reconstruct(model, dataset):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    save_folder = Path(f"./{model.config['task_name']}")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)
    
    save_name = 'reconstruction.jpg'
    save_file = save_folder / save_name
    
    idxs = torch.randint(0, len(dataset)-1, (100,))
    test_imgs = torch.cat([dataset[idx][0][None,:] for idx in idxs])
    
    model.eval()
    with torch.inference():
        
        test_imgs = test_imgs.to(device)
        
        recon_imgs, _, _, _ = model(test_imgs)
        
        test_imgs = (test_imgs + 1) / 2
        recon_imgs = (recon_imgs + 1) / 2
        if model.config['input_channels'] == 1:
            recon_imgs = 1 - recon_imgs
        
        out = torch.hstack([test_imgs, recon_imgs]) # Bx2CxWxH
        output = rearrange(out, "B (2 C) W H -> B C W (2 H)") #BxCxWx2H
        
        grid = torchvision.utils.make_grid(output, nrow=10)
        grid_img = torchvision.transforms.ToPILImage()(grid)
        grid_img.save(save_file)
        print(f"[INFO] The reconstructed image has been saved into {save_file}.")


def save_encodings(model, dataloader):
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    save_folder = Path("./{model.config['task_name']}")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)
    
    save_name = 'encoding.pkl'
    save_file = save_folder / save_name
    
    encodings = None
    
    model.eval()
    with torch.inference_mode():
        
        for batch, (X, y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)
            
            _, encoding, _, _ = model(X) # Bx8x8
             
            encodings = encoding if encodings == None else torch.cat([encodings, encoding], dim = 0) # (nxB) x 8 x 8
        
        pickle.dump(encodings, open(save_file, "wb"))
        print(f"[INFO] All the encodings have been saved into the {save_file}")
        
        
            