import os
import argparse
import logging
from tempfile import NamedTemporaryFile
import random
import json
from base64 import b64encode
# from tqdm import tqdm, trange

import math
import numpy as np
# import mlxu

import torch
import torch.nn.init as init

# import jax
# import jax.numpy as jnp
# import flax
from einops import rearrange
from scipy.ndimage import binary_erosion, generate_binary_structure

from PIL import Image
# from muse import VQGANModel
# from utils import read_image_to_tensor
from ACDCDataset import ACDCDataset
from utilities import image_utils, utils
from models.vqvae_muse import get_tokenizer_muse, VQGANModel
from monai.networks.utils import one_hot


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


def main(args):
    logging.basicConfig(
        filename=f'/homes/jkl223/Desktop/Individual Project/logs/test_VQGAN.log',    # Log file path
        filemode='w',                       # 'a' - append, 'w' - overwrite
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s - %(name)s - %(message)s'
    )
    logger.info(f'Device: {device}')            # print('Device: {0}'.format(device))
    
    logger.info(f'''
                ###########################################################################
                                    >>>   Tokenizing data   <<< 
                Saved model to run:                     VQGAN        
                Data to tokenize:                       {args.dataset}
                ###########################################################################
                ''')
    
    # Load the pre-trained vq model from the hub
    # net = VQGANModel.from_pretrained('vqlm/muse/ckpts/laion').to(device)
    net = get_tokenizer_muse('')
    pretrained_weights = net.state_dict()
    
    image_model = VQGANModel(num_channels=1)
    image_model = adapt_pretrained(pretrained_weights, image_model)
    mask_model = VQGANModel(num_channels=4)
    mask_model = adapt_pretrained(pretrained_weights, mask_model)
    
    net.to(device)
    image_model.to(device)
    mask_model.to(device)
    net.eval()
    image_model.eval()
    mask_model.eval()
           
    slice_size = (128, 128, 16)
    dataset_path = '/vol/bitbucket/jkl223/ACDC'
    if args.dataset == 'train':
        dataset = ACDCDataset(dataset_path, slice_size=slice_size, val=False, training=True)
        dataset_images, dataset_labels = dataset.images_train, dataset.labels_train
    elif args.dataset == 'test':
        dataset = ACDCDataset(dataset_path, slice_size=slice_size, val=False, training=False)
        dataset_images, dataset_labels = dataset.images_test, dataset.labels_test
        
    out_dir = '/homes/jkl223/Desktop/Individual Project/results/tokens'
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, f'{args.dataset}_tokens.jsonl')

    with torch.no_grad():
        with NamedTemporaryFile() as ntf:
            all_tokens = np.memmap(ntf, dtype='i4', mode='w+', shape=(len(dataset), slice_size[-1], 512))
            all_tokens[:] = 0

            index = 0
            for image, label in zip(dataset_images, dataset_labels):
                image, label = np.array(image), np.array(label)
                image, label = image_utils.resize_3D(image, slice_size), image_utils.resize_3D(label, slice_size)
                # utils.save_slice(image[..., image.shape[-1] // 2], label[..., label.shape[-1] // 2], name=f'original_')
                utils.save_image_and_label(image, label, name=f'original_')
                
                image = image[np.newaxis, ...]            # convert to (1, 256, 256, 16)
                image = rearrange(image, 'c h w d -> d c h w') # use depth as batch dimension
                
                label = label[np.newaxis, ...]
                label = rearrange(label, 'c h w d -> d c h w')    
                image, label = torch.from_numpy(image), torch.from_numpy(label)

                label = one_hot(label, num_classes=4)
                # label = label[:, 1:, ...]                          # remove background class
                
                _, input_tokens = image_model.encode(image.to(device, dtype=torch.float32))
                _, label_tokens = mask_model.encode(label.to(device, dtype=torch.float32))
                
                # shape of tokens - (slices, 256)
                
                reconstructed_image = image_model.decode_code(input_tokens)
                reconstructed_label = mask_model.decode_code(label_tokens)
                
                reconstructed_image = rearrange(reconstructed_image, 'd c h w -> c h w d')
                # reconstructed_image = torch.argmax(reconstructed_image, dim=0)
                reconstructed_label = rearrange(reconstructed_label, 'd c h w -> c h w d')
                # background = torch.zeros((1, *reconstructed_label.shape[1:]), dtype=reconstructed_label.dtype).to(device)
                # reconstructed_label = torch.cat((background, reconstructed_label), dim=0)
                reconstructed_label = torch.argmax(reconstructed_label, dim=0)
                
                reconstructed_image = reconstructed_image.cpu().numpy().squeeze()
                reconstructed_label = reconstructed_label.cpu().numpy().squeeze()
                
                reconstructed_label = remap_labels(reconstructed_label, {0: 0, 1: 3, 2: 2, 3: 1})
                # black, red, blue, green
                # ['black', 'green', 'blue', 'red']
                
                print(reconstructed_image.shape, reconstructed_label.shape)
                # utils.save_slice(reconstructed_image[..., reconstructed_image.shape[-1] // 2], reconstructed_label[..., reconstructed_label.shape[-1] // 2], name=f'reconstructed_')
                utils.save_image_and_label(reconstructed_image, reconstructed_label, name=f'reconstructed_')
                # utils.save_gt(reconstructed_label, name=f'reconstructed_gt_')
                
                return                
            
            # for input_image_batch, output_image_batch in tqdm(dataloader, ncols=0):
            #     '''b h w c -> b c h w'''
            #     _, input_token_batch = net.encode(input_image_batch.permute(0,3,1,2).to(device))
            #     _, output_token_batch = net.encode(output_image_batch.permute(0, 3, 1, 2).to(device))


            #     all_tokens[index:index + input_image_batch.shape[0]] = np.concatenate(
            #         [input_token_batch.cpu().numpy().astype(np.int32), output_token_batch.cpu().numpy().astype(np.int32)],
            #         axis=1
            #     )
            #     index += input_image_batch.shape[0]

            # with open(FLAGS.output_file, 'w') as fout:
            #     for _ in trange(FLAGS.n_epochs, ncols=0):
            #         indices = np.random.permutation(total_images).reshape(-1, FLAGS.n_shots)
            #         for i in trange(indices.shape[0], ncols=0):
            #             tokens = all_tokens[indices[i], :].reshape(-1)
            #             data = {'tokens': b64encode(tokens.tobytes()).decode('utf-8'),}
            #             fout.write(json.dumps(data) + '\n')


def remap_labels(predicted, mapping):
    remapped = np.copy(predicted)
    for original, new in mapping.items():
        remapped[predicted == original] = new
        
    for slc in range(remapped.shape[-1]):
        remapped[..., slc] = update_red_pixels(remapped[..., slc])
    return remapped


def update_red_pixels(image):
    """
    Update pixels classified as 3 (red) that touch pixels classified as 1 (green) to 1 (green).
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input image must be a NumPy array.")

    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    to_update = np.zeros_like(image, dtype=bool)  # Array to mark red pixels to update
    
    def flood_fill(start_row, start_col):
        queue = [(start_row, start_col)]
        visited = np.zeros_like(image, dtype=bool)
        while queue:
            row, col = queue.pop(0)
            if row < 0 or row >= image.shape[0] or col < 0 or col >= image.shape[1]:
                continue
            if visited[row, col]:
                continue
            visited[row, col] = True
            
            if image[row, col] == 3:  # Only consider red pixels
                to_update[row, col] = True  # Mark red pixel for update

            for dr, dc in directions:
                neighbor_row, neighbor_col = row + dr, col + dc
                if 0 <= neighbor_row < image.shape[0] and 0 <= neighbor_col < image.shape[1]:
                    if image[neighbor_row, neighbor_col] == 3 and not visited[neighbor_row, neighbor_col]:
                        queue.append((neighbor_row, neighbor_col))
    
    # Start flood fill from each green pixel
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if image[row, col] == 1:
                flood_fill(row, col)
    
    # Update the red pixels that were marked
    image[to_update] = 1

    return image


def adapt_pretrained(pretrained_weights, new_model):
    new_model_state_dict = new_model.state_dict()
    # Adapt the first convolutional layer in the Encoder
    if new_model.config.num_channels != pretrained_weights['encoder.conv_in.weight'].shape[1]:
        old_conv_weight = pretrained_weights['encoder.conv_in.weight']
        
        if new_model.config.num_channels == 1:
            new_conv_weight = old_conv_weight.mean(dim=1, keepdim=True)
        elif new_model.config.num_channels == 4:
            # new_conv_weight = torch.cat([old_conv_weight, old_conv_weight[:, :1, :, :]], dim=1)
            # Randomly initialize the new weights for the additional channel
            new_conv_weight = torch.cat([old_conv_weight, torch.zeros_like(old_conv_weight[:, :1, :, :])], dim=1)
            # init.kaiming_uniform_(new_conv_weight[:, -1, :, :], a=math.sqrt(5))  # He initialization for the new channel
            new_conv_weight[:, -1, :, :] = torch.mean(new_conv_weight[:, 1:3, :, :], dim=1)
        else:
            raise ValueError("Unsupported number of channels")

        new_model_state_dict['encoder.conv_in.weight'] = new_conv_weight

    # Adapt the last convolutional layer in the Decoder
    if new_model.config.num_channels != pretrained_weights['decoder.conv_out.weight'].shape[0]:
        old_conv_weight = pretrained_weights['decoder.conv_out.weight']
        
        if new_model.config.num_channels == 1:
            new_conv_weight = old_conv_weight.mean(dim=0, keepdim=True)
        elif new_model.config.num_channels == 4:
            # new_conv_weight = torch.cat([old_conv_weight, old_conv_weight[:1, :, :, :]], dim=0)
            new_conv_weight = torch.cat([old_conv_weight, torch.zeros_like(old_conv_weight[:1, :, :, :])], dim=0)
            # init.kaiming_uniform_(new_conv_weight[-1, :, :, :], a=math.sqrt(5))
            new_conv_weight[-1, :, :, :] = torch.mean(new_conv_weight[1:3, :, :, :], dim=0)
        else:
            raise ValueError("Unsupported number of channels")

        new_model_state_dict['decoder.conv_out.weight'] = new_conv_weight
    
    # Adapt the last convolutional layer in the Decoder (biases)
    if new_model.config.num_channels != pretrained_weights['decoder.conv_out.bias'].shape[0]:
        old_conv_bias = pretrained_weights['decoder.conv_out.bias']
        
        if new_model.config.num_channels == 1:
            new_conv_bias = old_conv_bias.mean(dim=0, keepdim=True)
        elif new_model.config.num_channels == 4:
            # new_conv_bias = torch.cat([old_conv_bias, old_conv_bias[:1]], dim=0)
            new_conv_bias = torch.cat([old_conv_bias, torch.zeros_like(old_conv_bias[:1])], dim=0)
            # init.uniform_(new_conv_bias[-1], a=-1, b=1)
            new_conv_bias[-1] = torch.mean(new_conv_bias[1:3], dim=0)
        else:
            raise ValueError("Unsupported number of channels for decoder bias")

        new_model_state_dict['decoder.conv_out.bias'] = new_conv_bias
    
    # Load the adapted state_dict into the new model
    for key in new_model_state_dict.keys():
        if key in pretrained_weights and key not in ['encoder.conv_in.weight', 'decoder.conv_out.weight', 'decoder.conv_out.bias']:
            new_model_state_dict[key] = pretrained_weights[key]

    new_model.load_state_dict(new_model_state_dict)
    
    return new_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tokenisation parameters')
    parser.add_argument('dataset', type=str, choices=['train', 'test'], help='Dataset type to tokenise')
    parser.add_argument('-t', '--test', action='store_true', help='Use test params (decode and get dice score after tokenisation)')
    args = parser.parse_args()
    
    main(args)