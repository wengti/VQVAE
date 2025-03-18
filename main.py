
import torch
import torch.nn as nn
from load_data import custom_dataset
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model import VQVAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from engine import train_step, test_step
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path

# Setup
torch.manual_seed(111)
torch.cuda.manual_seed(111)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



configPath1 = './no_colour_5_cb_2_lat.yaml'
configPath2 = './color_20_cb_8_lat.yaml'

with open(configPath2, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

for key in config.keys():
  print(f"{key}: {config[key]}")


# 1. Load dataset
trainData = custom_dataset(directory = './data/train',
                           config = config)

testData = custom_dataset(directory = './data/test',
                          config = config)

# 2. Visualize dataset

randNum = torch.randint(0, len(trainData)-1, (9,))

for idx, num in enumerate(randNum):
    
    trainImg, trainLabel = trainData[num]
    trainImgPlt = ((trainImg+1)/2).permute(1,2,0)
    
    plt.subplot(3,3,idx+1)
    color = None if config['input_channels'] == 3 else 'gray'
    plt.imshow(trainImgPlt, cmap=color)
    plt.title(f"Label: {trainLabel}")
    plt.axis(False)

plt.tight_layout()
plt.show()
    

print(f"[INFO] Total number of images in the dataset: {len(trainData)}")
print(f"[INFO] Size of an image                     : {trainImg.shape}")
print(f"[INFO] Range of values within an image      : {trainImg.min()} to {trainImg.max()}")
print(f"[INFO] Available classes                    : {trainData.classes}")

# 3. Create dataloader
trainDataLoader = DataLoader(dataset = trainData,
                             batch_size = config['batch_size'],
                             shuffle = True)

testDataLoader = DataLoader(dataset = testData,
                            batch_size = config['batch_size'],
                            shuffle = False)

# 4. Visualize dataloader
trainImgBatch, trainLabelBatch = next(iter(trainDataLoader))

print(f"[INFO] Total number of batches    : {len(trainDataLoader)}")
print(f"[INFO] Number of images per batch : {trainImgBatch.shape[0]}")
print(f"[INFO] Shape of an image          : {trainImgBatch[0].shape}")

# 5. Create model
model0 = VQVAE(config = config).to(device)

# 6. Verify model
# =============================================================================
# from torchinfo import summary
# 
# summary(model = model0,
#         input_size = (1,1,28,28),
#         col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
#         row_settings = ['var_names'])
# =============================================================================


# 7. Optimizer and Scheduler

optimizer = torch.optim.Adam(params = model0.parameters(),
                             lr = config['learning_rate'])

scheduler = ReduceLROnPlateau(optimizer = optimizer,
                              factor = 0.5,
                              patience = 1)

# 8. Create training loop

trainLossList = []
testLossList = []

bestLoss = np.inf

save_folder = Path(f"./{config['task_name']}")
if not save_folder.is_dir():
    save_folder.mkdir(parents = True,
                      exist_ok = True)

save_model_name = "best.pt"
save_model_file = save_folder / save_model_name



for epoch in tqdm(range(config['epochs'])):
    
    trainResult = train_step(model = model0,
                             dataloader = trainDataLoader,
                             device = device,
                             optimizer = optimizer,
                             config = config)
    
    testResult = test_step(model = model0,
                           dataloader = testDataLoader,
                           device = device,
                           config = config)
    
    trainLossList.append(trainResult['loss'].item())
    testLossList.append(testResult['loss'].item())
    
    scheduler.step(trainResult['loss'])
    
    print(f"Current epoch         :{epoch}")
    print(f"Train Loss            :{trainResult['loss']:.4f}")
    print(f"Test Loss             :{testResult['loss']:.4f}")
    
    if trainResult['loss'] < bestLoss:
        print(f"[INFO] The best loss has been improved from {bestLoss:.4f} to {trainResult['loss']:.4f}.")
        print(f"[INFO] Proceed to save this current as the best model at {save_model_file}.")
        bestLoss = trainResult['loss']
        torch.save(obj = model0.state_dict(),
                   f = save_model_file)
    