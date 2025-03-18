
import torch
import torch.nn as nn
from einops import einsum

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class encoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        act_map = {'leaky': nn.LeakyReLU(),
                   'tanh': nn.Tanh(),
                   'none': nn.Identity()}
        
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels = config['enc_conv_channels'][i], out_channels = config['enc_conv_channels'][i+1],
                          kernel_size = config['enc_conv_kernel_size'][i], stride = config['enc_conv_stride'][i],
                          padding = 1),
                nn.BatchNorm2d(num_features = config['enc_conv_channels'][i+1]),
                act_map[config['enc_conv_act']]
                ) for i in range(len(config['enc_conv_kernel_size'])-1)])
        
        last_idx = len(config['enc_conv_kernel_size']) - 1
        self.encoder_blocks.append(nn.Sequential(
            nn.Conv2d(in_channels = config['enc_conv_channels'][last_idx], out_channels = config['enc_conv_channels'][last_idx + 1],
                      kernel_size = config['enc_conv_kernel_size'][last_idx], stride = config['enc_conv_stride'][last_idx],
                      padding = 1)
            ))
        
    def forward(self, x):    
        out = x
        for block in self.encoder_blocks:
            out = block(out)
        return out


class decoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        act_map = {'leaky': nn.LeakyReLU(),
                   'tanh': nn.Tanh(),
                   'none': nn.Identity()}
        
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels = config['dec_tconv_channels'][i], out_channels = config['dec_tconv_channels'][i+1],
                                   kernel_size = config['dec_tconv_kernel_size'][i], stride = config['dec_tconv_stride'][i]),
                nn.BatchNorm2d(num_features = config['dec_tconv_channels'][i+1]),
                act_map[config['dec_tconv_act']]
                ) for i in range(len(config['dec_tconv_kernel_size']) - 1)])
        
        last_idx = len(config['dec_tconv_kernel_size']) - 1
        self.decoder_blocks.append(nn.Sequential(
            nn.ConvTranspose2d(in_channels = config['dec_tconv_channels'][last_idx], out_channels = config['dec_tconv_channels'][last_idx+1],
                               kernel_size = 4, stride = 1),
            nn.Tanh()
            ))
    
    def forward(self, x):
        out = x
        for block in self.decoder_blocks:
            out = block(out)
        return out


class quantizer(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        
        self.codebook = nn.Embedding(num_embeddings = config['codebook_size'],
                                     embedding_dim = config['latent_dim']).to(device) # cb x lat, float tensor, same device
        
        self.config = config
        
    def forward(self, x):
        
        B, lat, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # B x H x W x lat
        x = x.reshape((B, -1, x.shape[-1])) # B x (HxW) x lat
        
        repeated_codebook = self.codebook.weight[None,:].repeat((B, 1, 1)) # B x cb x lat
        dist = torch.cdist(x, repeated_codebook) # B x (HxW) x cb
        min_enc_idx = torch.argmin(dist, dim=-1) # B x (HxW)
        
        # out_quant - shortest distance vector
        # x_comp - raw input
        min_enc_idx_flat = min_enc_idx.reshape((-1)) # (BxHxW)
        out_quant = torch.index_select(self.codebook.weight, dim=0, index = min_enc_idx_flat) # (BxHxW) x lat
        x_comp = x.reshape((-1, lat)) # (BxHxW) x lat
        
        # Compute loss
        codebook_loss = torch.mean( (x_comp.detach() - out_quant) ** 2)
        commitment_loss = torch.mean( (x_comp - out_quant.detach()) ** 2)
        
        # Reparameterization trick
        out_quant = (out_quant - x_comp).detach() + x_comp # (BxHxW) x lat
        
        # Reshape for output
        out_quant = out_quant.reshape((B, H, W, lat)) # B x H x W x lat
        out_quant = out_quant.permute(0, 3, 1, 2) # B x lat x H x W
        
        # Reshape for min_enc_idx
        min_enc_idx = min_enc_idx.reshape((B, H, W)) # B x H x W
        
        return out_quant, min_enc_idx, codebook_loss, commitment_loss
    
    def quantize_codebook_index(self , index):
        """
            index - integer tensor, one hot encoded, shape of b c h w, same device as tensor
        """
        
        out = einsum(index, self.codebook.weight, 'b n h w, n d -> b d h w') # B x lat x H x W
        return out


class VQVAE(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.encoder = encoder(config = config)
        self.pre_conv = nn.Conv2d(in_channels = config['enc_conv_channels'][-1], out_channels = config['latent_dim'],
                                 kernel_size = 1, stride = 1)
        
        self.quantizer = quantizer(config = config)
        
        self.post_conv = nn.Conv2d(in_channels = config['latent_dim'], out_channels = config['dec_tconv_channels'][0],
                                   kernel_size = 1, stride = 1)
        self.decoder = decoder(config = config)
    
    def forward(self, x):
        out = x
        out = self.encoder(out)
        out = self.pre_conv(out)
        out, min_enc_idx, codebook_loss, commitment_loss = self.quantizer(out)
        out = self.post_conv(out)
        out = self.decoder(out)
        
        return out, min_enc_idx, codebook_loss, commitment_loss
    
    def generate_from_index(self, index):
        """
            index - integer tensor, one hot encoded, shape of b c h w, same device as tensor
        """
        
        out = self.quantizer.quantize_codebook_index(index = index) # B x lat x H x W
        out = self.post_conv(out)
        out = self.decoder(out)
        return out
        