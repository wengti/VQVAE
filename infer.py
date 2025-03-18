
import torch
from einops import rearrange
import torchvision
import pickle
from pathlib import Path

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def reconstruct(model, dataset, config):
    
    save_folder = Path(f"./{config['task_name']}")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)
    
    save_name = "reconstruction.jpg"
    save_file = save_folder / save_name
    
    idxs = torch.randint(0, len(dataset)-1, (100,))
    test_imgs = torch.cat([dataset[idx][0][None, :] for idx in idxs], dim = 0).to(device) # 100 x C x H x W
    
    model.eval()
    with torch.inference_mode():
        
        recon_imgs, _, _, _ = model(test_imgs) # 100 x C x H x W
        
        test_imgs = (test_imgs + 1) / 2
        recon_imgs = (recon_imgs + 1) / 2
        if config['input_channels'] == 1:
            recon_imgs = 1 - recon_imgs
        
        out = torch.hstack([test_imgs, recon_imgs]) # 100 x 2C x H x W
        output = rearrange(out, 'B (d C) H W -> B C H (d W)', d = 2) # 100 x C x H x 2W
        grid = torchvision.utils.make_grid(output, nrow=10)
        output_img = torchvision.transforms.ToPILImage()(grid)
        output_img.save(save_file)
        print(f"[INFO] The reconstructed image has been saved into {save_file}.")


def save_encodings(model, dataloader, config):
    
    min_enc_idx_list = []
    
    save_folder = Path(f"./{config['task_name']}")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)
    
    save_name = "encodings.pkl"
    save_file = save_folder / save_name
    
    model.eval()
    with torch.inference_mode():
        
        for batch, (X,y) in enumerate(dataloader):
            
            X, y = X.to(device), y.to(device)
            _, min_enc_idx, _, _ = model(X) # B x H x W
            min_enc_idx_list.append(min_enc_idx) 
        
        min_enc_idx_list = torch.cat(min_enc_idx_list, dim=0) # [ BxHxW, BxHxW ] -> nB x H x W
        pickle.dump(min_enc_idx_list, open(save_file, "wb"))
        print(f"[INFO] The closest encoding index of the dataset has been saved into {save_file}.")
        


# =============================================================================
# reconstruct(model = model0,
#             dataset = testData,
#             config = config)
# 
# save_encodings(model = model0,
#                dataloader = trainDataLoader,
#                config = config)
# =============================================================================
        
        
