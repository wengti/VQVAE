
import torch
import torch.nn as nn
from einops import einsum

device = "cuda" if torch.cuda.is_available() else "cpu"

class encoder(nn.Module):

    def __init__(self, config):
        super().__init__()

        act_map = {'tanh': nn.Tanh(),
                   'leaky': nn.LeakyReLU(),
                   'none': nn.Identity()}

        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels = config['enc_conv_channels'][i], out_channels = config['enc_conv_channels'][i+1],
                          kernel_size = config['enc_conv_kernel_size'][i], stride = config['enc_conv_stride'][i],
                          padding = 1),
                nn.BatchNorm2d(num_features = config['enc_conv_channels'][i+1]),
                act_map[config['enc_conv_act']]
                ) for i in range(len(config['enc_conv_kernel_size']) - 1)])


        last_idx = len(config['enc_conv_stride']) - 1
        self.encoder_blocks.append(nn.Sequential(nn.Conv2d(in_channels = config['enc_conv_channels'][last_idx],
                                                           out_channels = config['enc_conv_channels'][last_idx + 1],
                                                           kernel_size = config['enc_conv_kernel_size'][last_idx],
                                                           stride = config['enc_conv_stride'][last_idx],
                                                           padding = 1)))

    def forward(self, x):
        out = x
        for block in self.encoder_blocks:
            out = block(out)
        return out


class decoder(nn.Module):

    def __init__(self, config):

        super().__init__()

        act_map = {'tanh': nn.Tanh(),
                   'leaky': nn.LeakyReLU(),
                   'none': nn.Identity()}

        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(in_channels = config['dec_tconv_channels'][i], out_channels = config['dec_tconv_channels'][i+1],
                                   kernel_size = config['dec_tconv_kernel_size'][i], stride = config['dec_tconv_stride'][i]),
                nn.BatchNorm2d(num_features = config['dec_tconv_channels'][i+1]),
                act_map[config['dec_tconv_act']]
                ) for i in range(len(config['dec_tconv_kernel_size']) - 1)])

        last_idx = len(config['dec_tconv_kernel_size']) - 1
        self.decoder_blocks.append(nn.Sequential(nn.ConvTranspose2d(in_channels = config['dec_tconv_channels'][last_idx],
                                                                    out_channels = config['dec_tconv_channels'][last_idx + 1],
                                                                    kernel_size = config['dec_tconv_kernel_size'][last_idx],
                                                                    stride = config['dec_tconv_stride'][last_idx]),
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
                                     embedding_dim = config['latent_dim']).to(device) # CB x lat

    def forward(self, x):
        B, lat, H, W = x.shape
        x = x.permute(0, 2, 3, 1) # BxHxWx lat
        x = x.reshape((B, -1, lat)) # B x (HxW) x lat

        repeated_codebook = self.codebook.weight[None,:].repeat((B,1,1)) # B x CB x lat
        dist = torch.cdist(x, repeated_codebook) # B x (HxW) x CB

        min_enc_idx = torch.argmin(dist, dim=-1) # B x (HxW)

        quant_out = torch.index_select(input = self.codebook.weight,
                                       dim = 0,
                                       index = min_enc_idx.reshape((-1))).to(device) # (BxHxW) x lat

        x = x.reshape((-1, lat)) # (BxHxW) x lat

        codebook_loss = torch.mean( (x.detach() - quant_out) ** 2)
        commitment_loss = torch.mean( (x - quant_out.detach()) ** 2)

        quant_out = (quant_out - x).detach() + x # (BxHxW) x lat

        quant_out = quant_out.reshape((B, H, W, lat)).permute(0, 3, 1, 2) # B x lat x H x W
        min_enc_idx = min_enc_idx.reshape((B, H, W)) # B x H x W

        return quant_out, min_enc_idx, codebook_loss, commitment_loss

    def quantize_index(self, indices):
        """
            indices
             - need to be tensor, integer, on same device as model
             - need to be one hot encoded in the n axis

        """
        return einsum(indices, self.codebook.weight, "B N H W, N D -> B D H W")


class VQVAE(nn.Module):
    def __init__(self, config):

        super().__init__()

        self.config = config

        self.encoder = encoder(config)

        self.pre_quant = nn.Conv2d(in_channels = config['enc_conv_channels'][-1],
                                   out_channels = config['latent_dim'],
                                   kernel_size =  1,
                                   stride = 1)


        self.quantizer = quantizer(config)

        self.post_quant = nn.Conv2d(in_channels = config['latent_dim'],
                                    out_channels = config['dec_tconv_channels'][0],
                                    kernel_size = 1,
                                    stride = 1)

        self.decoder = decoder(config)


    def forward(self, x):
        out = x
        out = self.encoder(out)
        out = self.pre_quant(out)
        out, min_enc_idx, codebook_loss, commitment_loss = self.quantizer(out)
        out = self.post_quant(out)
        out = self.decoder(out)

        return out, min_enc_idx, codebook_loss, commitment_loss


    def decode_from_codebook_index(self, indices):
        """
            indices, tensor, integer, same device as the model
            one hot encoded
        """

        quant_out = self.quantizer.quantize_index(indices)
        quant_out = self.post_quant(quant_out)
        quant_out = self.decoder(quant_out)
        return quant_out