import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sst.bidirectional_transformer import BidirectionalTransformer
from models.VQGAN import VQGAN


class Config:
    ''' Configurations for SST '''
    name = 'SST'
    slice_size = (256, 256, 16) # - needs to be divisible by (16, 16, 16)
    
    latent_dim = 256
    num_codebook_vectors = 1024
    beta = 0.25                 # - commitment loss scalar for the codebook
    image_channels = 1          # - number of channels of images
    vqgan_path = '/homes/jkl223/Desktop/Individual Project/models/checkpoints/VQGAN_1_5000.pt'

    batch_size = 4           #!!!
    loss_type = 'crossentropy'
    optimizer = 'adam'
    learning_rate = 6.68281460865691e-06 #2.25e-05
    beta1 = 0.7685 #0.9 #0.5
    beta2 = 0.94135956 #0.96
    weight_decay = 0.0003510902985039575 #4.5e-2
    warmup_epochs = 100      #!!!
    start_from_epoch = 0
    accum_grad = 10

    sos_token = 0
    n_layers = 12 #24
    n_heads = 4 #8
    dim = 1024                  # must be divisible by n_heads
    hidden_dim = 6144 #dim*4
    num_image_tokens = 256
    temperature = 1.2
    
    validation_freq = 100   #!!!


class SST(nn.Module):               # Sequential Segmentation Transformer
    ''' Architecture for SST '''

    def __init__(self, args):
        super().__init__()
        self.num_image_tokens = args.num_image_tokens
        self.sos_token = args.sos_token #args.num_codebook_vectors + 1
        self.mask_token_id = args.num_codebook_vectors
        self.temperature = args.temperature
        self.device = args.device

        self.gamma = self.gamma_func("cosine")

        self.transformer = BidirectionalTransformer(args)
        self.vqgan = VQGAN(args).load_checkpoint(args.vqgan_path).eval()
        # print(f"Transformer parameters: {sum([p.numel() for p in self.transformer.parameters()])}")
        '''
        self.vqgan = VQGAN(args).load_checkpoint(args.vqgan_path).eval()
        # Embedding layer to transform input tokens
        input_dim = args.num_image_tokens
        hidden_dim = args.dim
        self.embedding = nn.Embedding(args.num_codebook_vectors+2, hidden_dim)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, args.num_codebook_vectors)  # Output layer
        
        # Activation function
        self.relu = nn.ReLU()
        '''


    @torch.no_grad()
    def encode_to_z(self, x):
        quant_z, indices, _ = self.vqgan.encode(x)
        indices = indices.view(quant_z.shape[0], -1)
        return quant_z, indices
    
    
    def indices_to_image(self, indices, p1=16, p2=16, is_mask=False):
        ix_to_vectors = self.vqgan.codebook.embedding(indices).reshape(indices.shape[0], p1, p2, 256)
        ix_to_vectors = ix_to_vectors.permute(0, 3, 1, 2)
        image = self.vqgan.decode(ix_to_vectors, is_mask=is_mask)
        return image


    def forward(self, image, label, pkeep=0.5):
        _, z_indices = self.encode_to_z(image)
        _, z_label = self.encode_to_z(label)
        
        sos_tokens = torch.ones(image.shape[0], 1, dtype=torch.long, device=z_indices.device) * self.sos_token

        # mask = torch.bernoulli(pkeep * torch.ones(z_indices.shape, device=z_indices.device))
        # mask = mask.round().to(dtype=torch.int64)
        # random_indices = torch.randint_like(z_indices, self.transformer.num_codebook_vectors)
        # new_indices = mask * z_indices + (~mask) * random_indices
        # new_indices = torch.cat((sos_tokens, new_indices), dim=1)
        '''
        r = math.floor(self.gamma(np.random.uniform()) * z_indices.shape[1])
        sample = torch.rand(z_indices.shape, device=z_indices.device).topk(r, dim=1).indices
        mask = torch.zeros(z_indices.shape, dtype=torch.bool, device=z_indices.device)
        mask.scatter_(dim=1, index=sample, value=True)

        masked_indices = self.mask_token_id * torch.ones_like(z_indices, device=z_indices.device)
        masked_image = mask * z_indices + (~mask) * masked_indices

        masked_image = torch.cat((sos_tokens, masked_image), dim=1)
        '''
        masked_image = torch.cat((sos_tokens, z_indices), dim=1)
        # target = z_label
        target = torch.cat((sos_tokens, z_label), dim=1)
        # print(f'New indices shape: {new_indices.shape}')
        
        # logits = self.transformer(new_indices)
        logits = self.transformer(masked_image)
        
        return logits, target
        '''
        # Pass through embedding layer
        x = self.embedding(z_indices)
        # Apply first fully connected layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        
        # Output layer: produces logits for each codebook entry
        x = self.fc5(x)
        
        return x, z_label
        '''
        

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        if k == 0:
            out[:, :] = -float('inf') #self.sos_token
        else:
            out[out < v[..., [-1]]] = -float('inf') #self.sos_token
        return out


    def gamma_func(self, mode="cosine"):
        if mode == "linear":
            return lambda r: 1 - r
        elif mode == "cosine":
            return lambda r: np.cos(r * np.pi / 2)
        elif mode == "square":
            return lambda r: 1 - r ** 2
        elif mode == "cubic":
            return lambda r: 1 - r ** 3
        else:
            raise NotImplementedError


    def create_input_tokens_normal(self, num, label=None):
        # label_tokens = label * torch.ones([num, 1])
        # Shift the label by codebook_size
        # label_tokens = label_tokens + self.vqgan.codebook.num_codebook_vectors
        # Create blank masked tokens
        blank_tokens = torch.ones((num, self.num_image_tokens), device=self.device)
        masked_tokens = self.mask_token_id * blank_tokens
        # Concatenate the two as input_tokens
        # input_tokens = torch.concat([label_tokens, masked_tokens], dim=-1)
        # return input_tokens.to(torch.int32)
        return masked_tokens.to(torch.int64)


    def mask_by_random_topk(self, mask_len, probs, temperature=1.0):
        confidence = torch.log(probs) + temperature #* torch.distributions.gumbel.Gumbel(0, 1).sample(probs.shape).to(self.device)
        sorted_confidence, _ = torch.sort(confidence, dim=-1)
        # Obtains cut off threshold given the mask lengths.
        cut_off = torch.take_along_dim(sorted_confidence, mask_len.to(torch.long), dim=-1)
        # Masks tokens with lower confidence.
        masking = (confidence < cut_off)
        return masking


    @torch.no_grad()
    def sample_good(self, inputs=None, num=1, T=11, pkeep=0.5, mode="cosine"):
        self.transformer.eval()
        N = self.num_image_tokens
        if inputs is None:
            inputs = self.create_input_tokens_normal(num)
        else:                                               # inputs (batch, 128) for half sampling
            # Randomly mask half of the tokens in the input
            batch_size, seq_len = inputs.shape
            mask = torch.bernoulli(torch.full((batch_size, seq_len), pkeep, device=self.device)).bool()
            inputs = torch.where(mask, self.mask_token_id, inputs)
            
            # Pad input tokens if necessary
            if seq_len < N:
                inputs = torch.hstack(
                    (inputs, torch.zeros((batch_size, N - seq_len), device=self.device, dtype=torch.long).fill_(self.mask_token_id))
                )
            
        sos_tokens = torch.ones(inputs.shape[0], 1, dtype=torch.long, device=inputs.device) * self.sos_token
        inputs = torch.cat((sos_tokens, inputs), dim=1)

        unknown_number_in_the_beginning = torch.sum(inputs == self.mask_token_id, dim=-1)
        gamma = self.gamma_func(mode)
        cur_ids = inputs  # [8, 257]
        for t in range(T):
            logits = self.transformer(cur_ids)  # call transformer to get predictions [8, 257, 1024]
            # sampled_ids = torch.distributions.categorical.Categorical(logits=logits).sample()
            
            probs = F.softmax(logits, dim=-1)  # convert logits into probs [8, 257, 1024]
            sampled_ids = torch.argmax(probs, dim=-1)      # get the most probable token [8, 257]
            
            unknown_map = (cur_ids == self.mask_token_id)  # which tokens need to be sampled -> bool [8, 257]
            sampled_ids = torch.where(unknown_map, sampled_ids, cur_ids)  # replace all -1 with their samples and leave the others untouched [8, 257]
            # cur_ids = sampled_ids
            # print(f'Sampled ids shape: {sampled_ids.shape}')
            
            ratio = 1. * (t + 1) / T  # just a percentage e.g. 1 / 12
            mask_ratio = gamma(ratio)

            selected_probs = torch.squeeze(torch.take_along_dim(probs, torch.unsqueeze(sampled_ids, -1), -1), -1)  # get probability for selected tokens in categorical call, also for already sampled ones [8, 257]
            _CONFIDENCE_OF_KNOWN_TOKENS = torch.Tensor([torch.inf]).to(self.device)
            selected_probs = torch.where(unknown_map, selected_probs, _CONFIDENCE_OF_KNOWN_TOKENS)  # ignore tokens which are already sampled [8, 257]
            # print(f'Selected probs shape: {selected_probs.shape}')
            
            mask_len = torch.unsqueeze(torch.floor(unknown_number_in_the_beginning * mask_ratio), 1)  # floor(256 * 0.99) = 254 --> [254, 254, 254, 254, ....]
            mask_len = torch.maximum(torch.zeros_like(mask_len), torch.minimum(torch.sum(unknown_map, dim=-1, keepdim=True)-1, mask_len))  # add -1 later when conditioning and also ones_like. Zeroes just because we have no cond token
            # max(1, min(how many unknown tokens, how many tokens we want to sample))

            # Adds noise for randomness
            masking = self.mask_by_random_topk(mask_len, selected_probs, temperature=self.temperature * (1. - ratio))
            # Masks tokens with lower confidence.
            cur_ids = torch.where(masking, self.mask_token_id, sampled_ids)
            # print((cur_ids == 8192).count_nonzero())

        self.transformer.train()
        return cur_ids[:, 1:]

    
    @torch.no_grad()
    def sample(self, x, temperature=1.0, top_k=100):
        self.transformer.eval()
        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=x.device) * self.sos_token
        x = torch.cat((sos_tokens, x), dim=1)
        print(x.shape)
        for k in range(x.shape[1]):
            logits = self.transformer(x)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)
            probs = F.softmax(logits, dim=-1)
            ix = torch.multinomial(probs, num_samples=1)
            print(f'x shape: {x.shape}, ix shape: {ix.shape}')
            # ix = torch.argmax(probs, dim=-1, keepdim=True)
            x = torch.cat((x, ix), dim=1)
        x = x[:, sos_tokens.shape[1]:]
        self.transformer.train()
        return x
    
    
    @torch.no_grad()
    def sample_all(self, x, temperature=1.0, top_k=100):
        self.transformer.eval()

        # Add start-of-sequence (SOS) token
        sos_tokens = torch.ones(x.shape[0], 1, dtype=torch.long, device=x.device) * self.sos_token
        x = torch.cat((sos_tokens, x), dim=1)  # x is now (batch_size, seq_len + 1)

        # Iterate over the sequence length
        for k in range(x.shape[1] - 1):  # Iterate up to the last token (excluding the one just appended)
            # Pass the current sequence through the transformer
            logits = self.transformer(x)  # logits is (batch_size, seq_len + 1, vocab_size)
            
            # Scale logits by temperature and get the logits for the current position
            logits = logits[:, k+1, :] / temperature

            # Apply top-k filtering if specified
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Get the most probable token at each position
            ix = torch.argmax(probs, dim=-1, keepdim=True)  # ix is (batch_size, 1)

            # Replace the token at the next position with the predicted token
            x[:, k+1] = ix.squeeze(-1)

        # Remove the SOS token before returning the sequence
        x = x[:, 1:]  # x is now (batch_size, seq_len)

        self.transformer.train()
        return x
    

    @torch.no_grad()
    def log_images(self, x, T=11, pkeep=0.5, mode="cosine", test=False):
        log = dict()

        _, z_indices = self.encode_to_z(x)
        
        if test:
            output_indices = self.sample_good(T=T, pkeep=pkeep, mode=mode)
            # output_indices = self.sample(z_indices, temperature=1.0, top_k=100)
            # output_indices = self.sample_all(z_indices, temperature=1.0, top_k=100)
        else:
            output_indices = self.sample_good(z_indices, T=T, pkeep=pkeep, mode=mode)
        rec = self.indices_to_image(output_indices, is_mask=True)
        
        log["input"] = x
        log["output"] = rec
        
        return log
        
        # # create new sample
        # index_sample = self.sample_good(mode=mode)
        # print(f"Index sample shape: {index_sample.shape}")
        # x_new = self.indices_to_image(index_sample)

        # # create a "half" sample
        # z_start_indices = z_indices[:, :z_indices.shape[1] // 2]
        # half_index_sample = self.sample_good(z_start_indices, mode=mode)
        # print(f"Half index sample shape: {half_index_sample.shape}")
        # x_sample = self.indices_to_image(half_index_sample)

        # # create reconstruction
        # x_rec = self.indices_to_image(z_indices)
        
        # output_indices = self.sample_good(z_indices, mode=mode)
        # print(f"Output indices shape: {output_indices.shape}")
        # rec = self.indices_to_image(output_indices)
        # log["output"] = rec

        # log["input"] = x
        # log["rec"] = x_rec
        # log["half_sample"] = x_sample
        # log["new_sample"] = x_new
        # return log, torch.concat((x, x_rec, x_sample, x_new))
        
