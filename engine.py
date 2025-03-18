
import torch
import torch.nn as nn

loss_fn = nn.MSELoss()

def train_step(model, dataloader, device, optimizer, config):
    
    train_loss = 0
    
    for batch, (X,y) in enumerate(dataloader):
        
        X, y = X.to(device), y.to(device)
        
        X_recon, _, codebook_loss, commitment_loss = model(X)
        recon_loss = loss_fn(X_recon, X)
        loss = config['recon_loss_weight'] * recon_loss \
                + config['codebook_loss_weight'] * codebook_loss \
                    + config['commitment_loss_weight'] * commitment_loss
        
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    
    return {'model_name': model.__class__.__name__,
            'loss': train_loss}

def test_step(model, dataloader, device, config):
    
    test_loss = 0
    
    model.eval()
    with torch.inference_mode():
        
        for batch, (X,y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            
            X_recon, _, codebook_loss, commitment_loss = model(X)
            recon_loss = loss_fn(X_recon, X)
            loss = config['recon_loss_weight'] * recon_loss \
                    + config['codebook_loss_weight'] * codebook_loss \
                        + config['commitment_loss_weight'] * commitment_loss
                        
            test_loss += loss
        
        test_loss /= len(dataloader)
        
        return {'model_name': model.__class__.__name__,
                'loss': test_loss}
            
        
        