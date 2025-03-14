
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from load_data import custom_dataset
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model import VQVAE
from engine import train_step, test_step
from tqdm.auto import tqdm
from torch.optim.schedule import ReduceLROnPlateau
import numpy as np

# 0. Setup
torch.manual_seed(111)
torch.cuda.manual_seed(111)

device = "cuda" if torch.cuda.is_available() else "cpu"

configPath1 = Path("./no_colour_5_codebook_size_2_latent_dim.yaml")
configPath2 = Path("./colour_20_codebook_size_8_latent_dim.yaml")
with open(configPath2, "r") as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

for key in config.keys():
    print(f"[INFO] {key}: {config[key]}")


# 1. Load dataset
trainPath = './data/train'
testPath = './data/test'


trainData = custom_dataset(directory = trainPath,
                           config = config)
testData = custom_dataset(directory = testPath,
                          config = config)

# 2. Visualize dataset
randNum = torch.randint(0, len(trainData)-1, (9,))

for idx, num in enumerate(randNum):
    trainImg, trainLabel = trainData[num]
    
    trainImgPlt = (trainImg + 1)/2
    trainImgPlt = trainImgPlt.permute(1,2,0)
    
    plt.subplot(3,3,idx+1)
    if config['input_channels'] == 1:
        plt.imshow(trainImgPlt, cmap="gray")
    else:
        plt.imshow(trainImgPlt)
    plt.title(f"Image {num}/Label: {trainLabel}")
    plt.axis(False)

plt.tight_layout()
plt.show()

print(f"[INFO] Number of images in the dataset: {len(trainData)}")
print(f"[INFO] Size of image                  : {trainImg.shape}")
print(f"[INFO] Range of values in the image   : {trainImg.min()} to {trainImg.max()}")
print(f"[INFO] Available classes              : {trainData.classes}")


# 3. Create data loader
trainDataLoader = DataLoader(dataset = trainData,
                             batch_size = config['batch_size'],
                             shuffle = True)

testDataLoader = DataLoader(dataset = testData,
                           batch_size = config['batch_size'],
                           shuffle = False)

# 4. Visualize data loader
trainImgBatch, trainLabelBatch = next(iter(trainDataLoader))

print(f"[INFO] Total number of batches        : {len(trainDataLoader)}")
print(f"[INFO] The number of images per batch : {trainImgBatch.shape[0]}")
print(f"[INFO] Size of an image               : {trainImgBatch[0].shape}")


# 5. Instantiate a model
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


# 7. Create Training Loop

optimizer = torch.optim.Adam(params = model0.parameters,
                             lr = config['learning_rate'])

scheduler = ReduceLROnPlateau(optimizer = optimizer,
                             factor = 0.5,
                             patience = 1)

trainLossList = []
testLossList = []

bestLoss  = np.inf


save_folder = Path(f"./{config['task_name']}")
if not save_folder.is_dir():
    save_folder.mkdir(parents = True,
                      exist_ok = True)

save_model_name = "best.pt"
save_file = save_folder / save_model_name


for epoch in tqdm(range(config['epochs'])):
    
    trainResult = train_step(model = model0,
                             dataloader = trainDataLoader,
                             device = device,
                             optimizer = optimizer)
    
    testResult = test_step(model = model0,
                           dataloader = testDataLoader,
                           device = device)
    
    scheduler.step(trainResult['loss'])
    
    trainLossList.append(trainResult['loss'])
    testLossList.append(testResult['loss'])
    
    print(f"[INFO] Current Epoch  : {epoch}")
    print(f"[INFO] Train Loss     : {trainResult['loss']:.4f}")
    print(f"[INFO] Test Loss      : {testResult['loss']:.4f}")
    
    if trainResult['loss'] < bestLoss:
        print(f"[INFO] The best training loss has been improved from {bestLoss} to {trainResult['loss']}.")
        print("[INFO] Proceed to saving this as the best model...")
        bestLoss = trainResult['loss']
        
        torch.save(obj = model0.state_dict(),
                   f = save_file)
    
        
        
        
    
    
    
