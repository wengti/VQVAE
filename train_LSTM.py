
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import random
import yaml
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np
import torchvision

# Setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Hyperparameter
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001



# Config
configPath1 = "./no_colour_5_cb_2_lat.yaml"
configPath2 = "./color_20_cb_8_lat.yaml"

with open(configPath2, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# Save folder
save_folder = Path(f"./{config['task_name']}")
if not save_folder.is_dir():
    save_folder.mkdir(parents = True,
                      exist_ok = True)

save_enc_name = "encodings.pkl"
save_enc_file = save_folder / save_enc_name


# 1. Create custom dataset for LSTM
class custom_dataset_LSTM(Dataset):
    
    def __init__(self, config, pkl_file):
        
        self.config = config
        self.enc_idx_batches = pickle.load(open(pkl_file, "rb")) # B x H x W
        self.context_size = 32
        
        self.start_token = config['codebook_size']
        self.pad_token = config['codebook_size']+1
        self.sents_list = self.load_sents()
    
    def load_sents(self):
        
        sents_list = []
    
        for batch in self.enc_idx_batches: # HxW
        
            if random.random() > 0.1:
                continue
            
            batch_flat = batch.reshape((-1)) # (HxW)
            start_token = torch.tensor([self.start_token]).to(device) # 1
            batch_flat = torch.cat([start_token, batch_flat]) # (1+HxW)
            
            for i in range(1, len(batch_flat)):
                
                # Adjust starting token to ensure context always has the same context size
                start = 0
                if i > self.context_size:
                    start = i - self.context_size
                
                # Assign the context token
                context = batch_flat[start:i]
                
                # Pad to the context size if necessary
                if len(context) < self.context_size:
                    context = torch.nn.functional.pad(context, (0, self.context_size - len(context) ), 'constant', self.pad_token) # 32
                    
                target = batch_flat[i] # 1
                
                sent = (context, target)
                sents_list.append(sent)
        
        # sent_list: [(32, 1), (32,1)]
        return sents_list
    
    def __len__(self):
        return len(self.sents_list)
    
    def __getitem__(self, index):
        return self.sents_list[index]
    


# 2. Load the LSTM dataset
trainDataLSTM = custom_dataset_LSTM(config = config,
                                    pkl_file = save_enc_file)




#### Separate here ####

# 3. Visualize the LSTM dataset
randNum = torch.randint(0, len(trainDataLSTM)-1, (5,))


print(f"[INFO] Total number of data available for training LSTM: {len(trainDataLSTM)}") # 384,000

for idx, num in enumerate(randNum):
    
    context, target = trainDataLSTM[num]
    print(f"[INFO] Random samples           :{idx}")
    print(f"[INFO] The context itself       :{context}")
    print(f"[INFO] The corresponding target :{target}")
    print(f"[INFO] The size of context      :{context.shape}")
    print("")
    
    

#  4. Create dataloader
trainDataLoaderLSTM = DataLoader(dataset = trainDataLSTM,
                                 batch_size = BATCH_SIZE,
                                 shuffle = True)

# 5. Visualize the dataloader
trainContextBatch, trainTargetBatch = next(iter(trainDataLoaderLSTM))

print(f"The number of batches            :{len(trainDataLoaderLSTM)}")
print(f"The number of samples in 1 batch :{trainContextBatch.shape[0]}")
print(f"The size of 1 context sample     :{trainContextBatch[0].shape}")
print(f"The size of 1 target sample      :{trainContextBatch[1].shape}")



# 6. Create custom LSTM models

class custom_LSTM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.embedding = nn.Embedding(num_embeddings = config['codebook_size'] + 2,
                                      embedding_dim = 2)
        
        self.rnn = nn.LSTM(input_size = 2,
                           hidden_size = 128,
                           num_layers = 2,
                           batch_first = True)
        
        self.fc = nn.Sequential(nn.Linear(in_features = 128,
                                          out_features = 128 //4 ),
                                nn.ReLU(),
                                nn.Linear(in_features = 128 // 4,
                                          out_features = config['codebook_size']))
    
    def forward(self, x):
        
        out = x.int() # B x 32
        out = self.embedding(out) # B x 32 x 2
        out, _ = self.rnn(out) # B x 32 x 128
        out = out[:, -1, :] # B x 128
        out = self.fc(out) # B x cb
        return out
    
    
# 7. Create LSTM models
modelLSTM = custom_LSTM(config = config).to(device)

# 8. Verify the models
# =============================================================================
# from torchinfo import summary
# 
# summary(model = modelLSTM,
#         input_size = (1,32),
#         col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
#         row_settings = ['var_names'])
# =============================================================================


#### Separate here #####


# 9. Optimizer and loss functions
optimizer = torch.optim.Adam(params = modelLSTM.parameters(),
                             lr = LEARNING_RATE)

lossFn = nn.CrossEntropyLoss()

bestLoss = np.inf

save_model_name = "bestLSTM.pt"
save_model_file = save_folder / save_model_name

# 10. Training loop

def train_step(model, dataloader, device, optimizer, loss_fn):

    train_loss = 0
    
    for batch, (context, target) in enumerate(dataloader):
        
        context, target = context.to(device), target.to(device)
        
        y_logits = model(context) # B x cb
        
        loss = loss_fn(y_logits, target)
        
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(trainDataLoaderLSTM)
    
    return {'model_name': model.__class__.__name__,
            'loss': train_loss}


trainLossListLSTM = []
for epoch in tqdm(range(EPOCHS)):
    
    trainResult = train_step(model = modelLSTM,
                             dataloader = trainDataLoaderLSTM,
                             device = device,
                             optimizer = optimizer,
                             loss_fn = lossFn)
    
    trainLossListLSTM.append(trainResult['loss'].item())
    
    print(f"[INFO] Current epoch: {epoch}")
    print(f"[INFO] Train Loss   : {trainResult['loss']:.4f}")
    
    if trainResult['loss'] < bestLoss:
        print(f"[INFO] The train loss has been improved from {bestLoss} to {trainResult['loss']:.4f}")
        print(f"[INFO] Proceed to save this as the best LSTM model in {save_model_file}.")
        bestLoss = trainResult['loss']
        torch.save(obj = modelLSTM.state_dict(),
                   f = save_model_file)


#### Separate here ######


# 11. Generate image

def generate_image(LSTM, VQVAE, pkl_file, config):
    
    save_folder = Path(f"./{config['task_name']}")
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)
    
    save_name = "generated_images.jpg"
    save_file = save_folder / save_name
    
    
    
    LSTM.eval()
    VQVAE.eval()
    with torch.inference_mode():
        
        num_imgs_to_generate = 100
        
        batch_enc_index = pickle.load(open(pkl_file, "rb")) # nB x H x W
        h = batch_enc_index.shape[-2]
        w = batch_enc_index.shape[-1]
        idx_to_generate = h*w
        
        context_size = 32
        
        out_list = []
        
        for _ in range(num_imgs_to_generate):
            out = torch.tensor([config['codebook_size']])[None,:].float().to(device) # 1x1, float torch tensor, same device as model
            
            for _ in range(idx_to_generate):
                
                out_token = out # 1x generated_token_size+1
                
                # Pad the token to the size of context size
                if out_token.shape[-1] < context_size:
                    gap = context_size - out_token.shape[-1]
                    out_token = torch.nn.functional.pad(out_token, (0, gap), 'constant', config['codebook_size']+1) # 1x32
                    
                # Crop the token to only the last 32 tokens
                elif out_token.shape[-1] >= context_size:
                    start = out_token.shape[-1] - context_size
                    out_token = out_token[:, start:]
                
                
                logits = LSTM(out_token) # 1 x lat
                prob = torch.nn.functional.softmax(logits, dim=-1) # 1 x lat
                pred_token = torch.multinomial(prob, 1) # 1 x 1

                out = torch.cat([out, pred_token], dim=-1) # 1 x generated_token_size+1
            
            out = out[:, 1:] # 1 x generated_token_size
            out = out.reshape(out.shape[0], h, w)
            out_list.append(out) # [1 x H x W, 1 x H x W...]
        
        out_idx = torch.cat(out_list, dim=0).long() # 100 x H x W
        out_idx_one_hot = torch.nn.functional.one_hot(out_idx, config['codebook_size']) # 100 x H x W x cb
        out_idx_one_hot = out_idx_one_hot.permute(0,3,1,2) # 100 x cb x h x w
        
        output_images = VQVAE.generate_from_index(out_idx_one_hot.float()) # 100 x C x H x W
        output_images = (output_images + 1) / 2
        
        
        grid = torchvision.utils.make_grid(output_images, nrow = 10)
        grid_images = torchvision.transforms.ToPILImage()(grid)
        grid_images.save(save_file)
        print(f"[INFO] The generated image has been successfully saved into {save_file}.")

# =============================================================================
# 
# generate_image(LSTM = modelLSTM,
#                VQVAE = model0,
#                pkl_file = save_enc_file,
#                config = config)
# =============================================================================
    