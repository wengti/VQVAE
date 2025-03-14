import torch
import torch.nn as nn
from torch.utils.data import Dataset
import yaml
import pickle
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision
from model import VQVAE

# 0. Setup
torch.manual_seed(111)
torch.cuda.manual_seed(111)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

configPath1 = "./no_colour_5_codebook_size_2_latent_dim.yaml"
configPath2 = "./colour_20_codebook_size_8_latent_dim.yaml"

with open(configPath1, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
        
save_folder = Path(f"./{config['task_name']}")
if not save_folder.is_dir():
    save_folder.mkdir(parents = True,
                      exist_ok = True)
    
pkl_name = 'encoding.pkl'    
pkl_file = save_folder / pkl_name

save_model_name = 'LSTM_best.pt'
save_model_file = save_folder / save_model_name


# Hyperparameter
BATCH_SIZE = config['batch_size']
EPOCHS = 10

        
# 1. Create LSTM Dataset
class custom_LSTM_dataset(Dataset):
    
    def __init__(self, config, pkl_file):
        
        self.start_token = config['codebook_size']
        self.pad_token = config['codebook_size']+1
        self.context_size = 32
        
        indices = pickle.load(open(pkl_file, "rb")) #nB x 8 x 8
        indices = indices.reshape((indices.shape[0], -1)) # nB x 64
        
        repeated_start_tokens = torch.tensor([self.start_token])[None,:].repeat((indices.shape[0],1)).to(device) #nB x 1
        indices = torch.cat([repeated_start_tokens, indices], dim=1) # nB x 65 
        
        self.indices = indices #nB x 65
        self.sents_list = self.load_sents()
        
    
    def load_sents(self):
        
        sents_list = []
        for row in self.indices:
            for i in range(1, len(row)):
                
                # Handle if the context size is too much
                start = 0
                if i > 32:
                    start = i - self.context_size
                
                # Obtain the context 
                context = row[start:i]
                
                # Pad the context size if it is less than the context size
                if len(context) < self.context_size:
                    gap = self.context_size - len(context)
                    context = torch.nn.functional.pad(context, (0,gap), "constant", self.pad_token)
                
                # Obtain the target
                target = row[i]
                
                sent = (context, target)
                sents_list.append(sent)
                
        return sents_list
    
    def __len__(self):
        return len(self.sents_list)
    
    def __getitem(self, index):
        context, target = self.sent_list[index]
        return context, target


# 2. Load dataset
trainDataLSTM = custom_LSTM_dataset(config = config,
                                    pkl_file = pkl_file)

# 3. Visualize dataset
print(f"[INFO] Number of available samples: {len(trainDataLSTM)}") # Expected 60,000 x 34

randNum = torch.randint(0, len(trainDataLSTM)-1, (5,))
for i, num in enumerate(randNum):
    print(f"[INFO] Random sample {i}          : {trainDataLSTM[num][0]}")
    print(f"[INFO] Size of the sample         : {len(trainDataLSTM[num][0])}")
    print(f"[INFO] The corresponding context  : {trainDataLSTM[num][1]}")
    
# 4. Create dataloader
trainDataLoaderLSTM = DataLoader(dataset = trainDataLSTM,
                                 batch_size = BATCH_SIZE,
                                 shuffle = True)

# 5. Create model
class custom_LSTM(nn.Module):
    
    def __init__(self, config):
        super().__init__()
            
        input_size = config['lstm_input_size']
        hidden_size = config['lstm_hidden_size']
        codebook_size = config['codebook_size']
        
        self.embedding = nn.Embedding(num_embeddings = codebook_size+2,
                                      embedding_dim = input_size)
        self.rnn = nn.LSTM(input_size = input_size,
                           hidden_size = hidden_size,
                           num_layers = 2,
                           batch_first = True)
        self.FC = nn.Sequential(nn.Linear(in_features = hidden_size,
                                          out_features = hidden_size // 4),
                                nn.ReLU(),
                                nn.Linear(in_features = hidden_size // 4,
                                          out_features = codebook_size))
    
    def forward(self, x):
        out = x.int() # B x 32
        
        out = self.embedding(out) # B x 32 x 2
        out, _ = self.rnn(out) # B x 32 x 128
        
        out = out[:, -1, :] # B x 128
        out = self.FC(out) # B x cb
        return out


# 6. Initialize the model
LSTM0 = custom_LSTM(config = config).to(device)

# 7. Verify the model
# =============================================================================
# from torchinfo import summary
# 
# summary(model = LSTM0,
#         input_size = (1,32),
#         col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
#         row_settings = ['var_names'])
# =============================================================================

# 8. Optimizer and loss function
optimizer = torch.optim.Adam(params = LSTM0.parameters(),
                             lr = 1e-3)

lossFn = nn.CrossEntropyLoss()

# 9. Create training step
def train_step(model, dataloader, device, optimizer, loss_fn):
    
    train_loss = 0
    
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_logits = model(X)
        
        loss = loss_fn(y_logits, y)
        
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    
    return {'model name': model.__class__.__name__,
            'loss': train_loss}

# 10. Create training loop
for epoch in tqdm(range(tqdm(EPOCHS))):
    
    trainResult = train_step(model = LSTM0,
                             dataloader = trainDataLoaderLSTM,
                             device = device,
                             optimizer = optimizer,
                             loss_fn = lossFn)
    
    print(f"[INFO] Current epoch: {epoch}")
    print(f"[INFO] Train Loss: {trainResult['loss']:.4f}")
    
# 11. Save models
torch.save(obj = LSTM0.state_dict(),
           f = save_model_file)




########### Separate here ###########







# 12. generate imgs

def generate_imgs(LSTM, VQVAE, config, pkl_file, device):
        
    num_samples = 100
    context_size = 32
    
    encodings = pickle.load(open(pkl_file, "rb")) #nB x 8 x 8 
    encoding_length = encodings.shape[-1] * encodings.shape[-2]
    
    out_idx_list= []
    
    save_folder = f"./{config['task_name']}"
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)
    
    save_img_name = "Generated Imgs.jpg"
    save_img_file = save_folder / save_img_name
    
    
    LSTM.eval()
    with torch.inference_mode():
        
        for _ in range(num_samples):
            out = torch.tensor([config['codebook_size']])[None,:] # 1x1, tensor, float
            
            for j in range(encoding_length):
                
                if out.shape[-1] < context_size:
                    out_pad = torch.nn.functional.pad(out, (0, context_size - out.shape[-1]), "constant", config['codebook_size']+1) # 1x32
                
                y_logits = LSTM(out_pad) # 1 x cb
                y_prob = torch.softmax(y_logits, dim=-1) # 1 x cb
                y_pred = torch.multinomial(y_prob[0], num_samples = 1) # 1
                
                y_pred = y_pred[None, :] # 1x1
                
                out = torch.cat([out, y_pred], dim=1) # 1x2 (1st)... 1 x 65 (last iterations)
            
            out_idx_list.append(out[:, 1:]) # 100 x 1 x 64
        
        out_idx_tensor = torch.cat(out_idx_list, dim=0) # 100 x 1 x 64
        
        h = (out_idx_tensor.shape[-1] ** 0.5).int()
        
        out_idx_tensor = out_idx_tensor.reshape((out_idx_tensor.shape[0], h, h)).int() # 100 x 8 x 8
        out_idx_tensor_one_hot = torch.nn.functional.one_hot(tensor = out_idx_tensor,
                                                             num_classes = config['codebook_size']) # 100 x 8 x 8 x cb
        out_idx_tensor_one_hot = out_idx_tensor_one_hot.permute(0,3,1,2) # 100 x cb x 8 x 8
        
        out_imgs = VQVAE.decode_from_codebook_index(out_idx_tensor_one_hot.to(device))
        
        out_imgs = (out_imgs + 1) / 2
        
        grid = torchvision.utils.make_grid(out_imgs, nrow = 10)
        recon_imgs = torchvision.transforms.ToPILImage()(grid)
        recon_imgs.save(save_img_file)
        print(f"All the images have been saved into {save_img_file}.")

# 13. Call the functions to generate imgs

save_folder = Path(f"./{config['task_name']}")
if not save_folder.is_dir():
    save_folder.mkdir(parents = True,
                      exist_ok = True)

save_lstm_name = 'LSTM_best.pt'
save_lstm_file = save_folder / save_lstm_name

save_vqvae_name = "best.pt"
save_vqvae_file = save_folder / save_vqvae_name

pkl_name = 'encoding.pkl'    
pkl_file = save_folder / pkl_name

LSTM1 = custom_LSTM(config = config)
LSTM1.load_state_dict(torch.load(f = save_lstm_file,
                                 weights_only = True)).to(device)

VQVAE1 = VQVAE(config = config)
VQVAE.load_state_dict(torch.load(f=save_vqvae_file,
                                 weights_only = True)).to(device)


generate_imgs(LSTM = LSTM1,
              VQVAE = VQVAE1,
              config = config,
              pkl_file = pkl_file,
              device = device)
        
        
        
        
                
            



        

    
            
             
            
        
        
            
            
        