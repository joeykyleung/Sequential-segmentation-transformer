import torch
import torch.nn as nn
from models.vqgan.encoder import Encoder
from models.vqgan.decoder import Decoder
from models.vqgan.codebook import Codebook


class Config:
    ''' Configurations for VQGAN '''
    name = 'VQGAN'
    slice_size = (256, 256, 16) # - needs to be divisible by (16, 16, 16)
    
    latent_dim = 256
    num_codebook_vectors = 1024
    beta = 0.25                 # - commitment loss scalar for the codebook
    image_channels = 1          # - number of channels of images

    batch_size = 5           #!!!
    loss_type = 'lpips'
    optimizer = 'adam'
    learning_rate = 2.25e-05
    beta1 = 0.5
    beta2 = 0.9

    disc_start = 5           #!!!
    disc_slow_down = 5       #!!!
    disc_factor = 1.
    l2_loss_factor = 1.         # - weighting factor for reconstruction loss
    perceptual_loss_factor = 1. # - weighting factor for perceptual loss
    
    validation_freq = 500    #!!!


class VQGAN(nn.Module):
    def __init__(self, args):
        super(VQGAN, self).__init__()
        self.encoder = Encoder(args).to(device=args.device)
        # self.decoder = Decoder(args).to(device=args.device)
        self.decoder_image = Decoder(args, is_mask=False).to(device=args.device)  # Decoder for images
        self.decoder_mask = Decoder(args, is_mask=True).to(device=args.device)   # Decoder for masks
        self.codebook = Codebook(args).to(device=args.device)
        self.quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)
        self.post_quant_conv = nn.Conv2d(args.latent_dim, args.latent_dim, 1).to(device=args.device)

    def forward(self, imgs, is_mask=False):
        encoded_images = self.encoder(imgs)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        quantized_codebook_mapping = self.post_quant_conv(codebook_mapping)
        # decoded_images = self.decoder(quantized_codebook_mapping)
        if is_mask:
            decoded_images = self.decoder_mask(quantized_codebook_mapping)
        else:
            decoded_images = self.decoder_image(quantized_codebook_mapping)

        return decoded_images, codebook_indices, q_loss

    def encode(self, x):
        encoded_images = self.encoder(x)
        quantized_encoded_images = self.quant_conv(encoded_images)
        codebook_mapping, codebook_indices, q_loss = self.codebook(quantized_encoded_images)
        return codebook_mapping, codebook_indices, q_loss

    def decode(self, z, is_mask=False):
        quantized_codebook_mapping = self.post_quant_conv(z)
        # decoded_images = self.decoder(quantized_codebook_mapping)
        if is_mask:
            decoded_images = self.decoder_mask(quantized_codebook_mapping)
        else:
            decoded_images = self.decoder_image(quantized_codebook_mapping)
        return decoded_images
    
    def calculate_lambda(self, nll_loss, g_loss, is_mask=False):
        # last_layer = self.decoder.model[-1]
        # last_layer_weight = last_layer.weight
        if is_mask:
            last_layer_weight = self.decoder_mask.model[-1].weight
        else:
            last_layer_weight = self.decoder_image.model[-1].weight
        
        nll_grads = torch.autograd.grad(nll_loss, last_layer_weight, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer_weight, retain_graph=True)[0]
        
        λ = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        λ = torch.clamp(λ, 0, 1e4).detach()
        
        return 0.8 * λ

    @staticmethod
    def adopt_weight(disc_factor, i, threshold, value=0.):
        if i < threshold:
            disc_factor = value
        return disc_factor

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        return self