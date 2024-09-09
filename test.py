import torch
import torch.nn as nn
import os
import glob
import logging
import numpy as np

import argparse
from ACDCDataset import ACDCDataset
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.networks.utils import one_hot
from utilities import utils
from utilities import image_utils
from collections import OrderedDict
from einops import rearrange


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


def main(args):
    model_path_pattern = f'/homes/jkl223/Desktop/Individual Project/models/checkpoints/{args.model}_*.pt'
    model_files = glob.glob(model_path_pattern)
    # models_path = '/homes/jkl223/Desktop/Individual Project/models/checkpoints'
    # models = os.listdir(models_path)
    # models.sort(key=lambda x: int(x.split('_')[1].split('.')[0]), reverse=True)
    
    if args.model == 'UNET2D':
        from models.UNet2D import UNet2D as model_class, Config
    if args.model == 'UNET3D':
        from models.UNet3D import UNet3D as model_class, Config
    if args.model == 'UNETR':
        from models.UNETR import UNETR as model_class, Config
    if args.model == 'VQGAN':
        from models.VQGAN import VQGAN as model_class, Config
    if args.model == 'SST':
        from models.SST import SST as model_class, Config
        
    logging.basicConfig(
        filename=f'/homes/jkl223/Desktop/Individual Project/logs/test_{Config.name}.log',    # Log file path
        filemode='w',                       # 'a' - append, 'w' - overwrite
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s - %(name)s - %(message)s'
    )
    logger.info(f'Device: {device}')            # print('Device: {0}'.format(device))
    Config.device = device
    
    logger.info(f'''
                ###########################################################################
                                    >>>   Starting testing   <<< 
                Saved model to be tested:               {args.model}        
                Number of samples:                      {args.test_set_size}
                ###########################################################################
                ''')
    
    input_channels = 1
    output_channels = 4
    if Config.name in ['VQGAN', 'SST']:
        model = model_class(Config)
    else:
        model = model_class(input_channels, output_channels)
    
    if Config.name == 'UNETR':
        model_path = '/homes/jkl223/Desktop/Individual Project/models/checkpoints/UNETR.model'
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        # if state dict comes form nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = OrderedDict()
        curr_state_dict_keys = list(model.state_dict().keys())
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        
    else:
        if args.model_postfix == '-1':
            if model_files:
                model_path = model_files[-1]  # Assuming the list is sorted and the last file is the latest
            else:
                logger.info('No model checkpoint found')
        else:
            model_path = f'/homes/jkl223/Desktop/Individual Project/models/checkpoints/{args.model}{args.model_postfix}.pt'
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Model file '{model_path}' not found.")
        
        model.load_state_dict(torch.load(model_path, weights_only=True))

    print(f'loading model from {model_path}')
    logger.info(f'Loading {args.model} model from {model_path}')
    
    dataset_path = '/vol/bitbucket/jkl223/ACDC'
    dataset = ACDCDataset(dataset_path, slice_size=Config.slice_size, val=False, training=False)
    # dataset = ACDCDataset(dataset_path, slice_size=Config.slice_size, val=True, training=True)
    
    save_dir = os.path.join('/homes/jkl223/Desktop/Individual Project/results', Config.name)
    os.makedirs(save_dir, exist_ok=True)
    
    # total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(total_params)
    test(dataset, model, Config.name, args.test_set_size, Config.slice_size, save_dir)
    
    
def test(dataset, model, model_name, test_set_size, slice_size, save_dir):   
    model = model.to(device)
    model.eval()
    
    images, labels = dataset.get_random_batch(test_set_size, val=False, training=False, augment=False)
    # images, labels = dataset.get_random_batch(test_set_size, val=False, training=False, augment=False, index=22)
    images, labels = torch.from_numpy(images), torch.from_numpy(labels)
    images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
    
    logger.info(f'running test data (batch size: {test_set_size}) through {model_name} model...')
    with torch.no_grad():
        if model_name == 'SST':
            for slc in range(0, images.shape[-1]):
                log = model.log_images(images[..., slc], test=True)
                utils.save_slice(image=log['input'].cpu().numpy()[0].transpose(1, 2, 0),
                                 label=labels[..., slc].cpu().numpy()[0].squeeze(), name=f'{model_name}_original_{slc+1}')
                output = log['output'].cpu().numpy()[0].transpose(1, 2, 0)
                output = np.nan_to_num(output, nan=0)
                output = np.round(output)
                min_output = np.min(output)
                if min_output < 0:
                    output = output - min_output
                print(f'output: {np.unique(output)}, {len(np.unique(output))}, {output.shape}')
                utils.save_slice(label=log['output'].cpu().numpy()[0].transpose(1, 2, 0), name=f'{model_name}_reconstructed_{slc+1}')
            return
        if model_name == 'UNET2D':
            masks = torch.zeros_like(labels)
            for slc in range(0, images.shape[-1]):
                test_logits = model(images[..., slc])
                p = torch.softmax(test_logits, dim=1)
                output = torch.argmax(p, dim=1)
                output = output[:, np.newaxis, ...]
                masks[..., slc] = output
        elif model_name == 'VQGAN':
            decoded_images = torch.zeros_like(images)
            decoded_labels = torch.zeros_like(labels)
            for slc in range(0, images.shape[-1]):
                imgs = images[..., slc]
                lbls = labels[..., slc]
                # get codebook indices
                _, img_codebook_indices, _ = model.encode(imgs)
                _, lbl_codebook_indices, _ = model.encode(lbls)
                # print(f'image | mapping: {img_codebook_mapping.shape}, indices: {img_codebook_indices.shape}')
                # print(f'label | mapping: {lbl_codebook_mapping.shape}, indices: {lbl_codebook_indices.shape}')
                # get codebook vectors to decode
                decoded_image_vec = model.codebook.embedding(img_codebook_indices).reshape(test_set_size, 16, 16, 256) # match decoder (16 x 16 patches, 256 latent dim)
                decoded_image_vec = decoded_image_vec.permute(0, 3, 1, 2)
                decoded_label_vec = model.codebook.embedding(lbl_codebook_indices).reshape(test_set_size, 16, 16, 256)
                decoded_label_vec = decoded_label_vec.permute(0, 3, 1, 2)
                # decode vectors to images
                decoded_images[..., slc] = model.decode(decoded_image_vec, is_mask=False)
                decoded_lbl = model.decode(decoded_label_vec, is_mask=True)
                decoded_labels[..., slc] = torch.argmax(decoded_lbl, dim=1, keepdim=True)
                
                # decoded_images[..., slc], _, _ = model(imgs, is_mask=False)
                # decoded_lbl, _, _ = model(lbls, is_mask=True)
                # decoded_labels[..., slc] = torch.argmax(decoded_lbl, dim=1)
                
        else:
            if model_name == 'UNETR':
                images_t = rearrange(images, 'b c h w d -> b c d h w')
                test_logits = model(images_t)[0]
                test_logits = rearrange(test_logits, 'b c d h w -> b c h w d')
            else:
                test_logits = model(images)
            p = torch.softmax(test_logits, dim=1)
            output = torch.argmax(p, dim=1)
            masks = output[:, np.newaxis, ...]
    
    num_classes = 4
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_image = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    dice_label = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hausdorff = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hausdorff_image = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hausdorff_label = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
    
    if model_name == 'VQGAN':
        dice_image(y_pred=decoded_images, y=images)
        dice_label(y_pred=decoded_labels, y=labels)
        hausdorff_image(y_pred=decoded_images, y=images)
        hausdorff_label(y_pred=decoded_labels, y=labels)
        logger.info(f'dice (image): {dice_image.aggregate().item()}, dice (label): {dice_label.aggregate().item()}')
        logger.info(f'hausdorff distance (image): {hausdorff_image.aggregate().item()}, hausdorff distance (label): {hausdorff_label.aggregate().item()}')
        print(f'dice (image): {dice_image.aggregate().item()}, dice (label): {dice_label.aggregate().item()}')
        print(f'hausdorff distance (image): {hausdorff_image.aggregate().item()}, hausdorff distance (label): {hausdorff_label.aggregate().item()}')
    
        masks = decoded_images    # to use for saving images
    else:
        y_pred = one_hot(masks, num_classes=num_classes)
        y = one_hot(labels, num_classes=num_classes)
        
        dice_metric(y_pred=y_pred, y=y)
        dice_test_loss = 1 - dice_metric.aggregate().item()
        hausdorff(y_pred=y_pred, y=y)
        hausdorff_test_loss = hausdorff.aggregate().item()
        
        logger.info(f'dice loss: {dice_test_loss}, hausdorff distance: {hausdorff_test_loss}')
        print(f'dice loss: {dice_test_loss}, hausdorff distance: {hausdorff_test_loss}')
        
    images_np = images.cpu().numpy()
    masks_np = masks.cpu().numpy()
    labels_np = labels.cpu().numpy()
    label_masks_np = decoded_labels.cpu().numpy() if model_name == 'VQGAN' else np.array([0]*len(images_np))
    
    for i, (image, mask, label, label_mask) in enumerate(zip(images_np, masks_np, labels_np, label_masks_np)):
        logger.info(f'saving images {i+1}/{test_set_size}...')
        logger.info(f'image: {image.shape}, mask: {mask.shape}')
        image, mask, label, label_mask = image.squeeze(), mask.squeeze(), label.squeeze(), label_mask.squeeze()
        '''
        if model_name == 'VQGAN':
            utils.save_image_and_label(image, label, save_dir=save_dir, name=f'{model_name}_{i+1} - original_')
            utils.save_image_and_label(mask, label_mask, save_dir=save_dir, name=f'{model_name}_{i+1} - reconstructed_')
        else:
            utils.save_image_and_label(image, mask, save_dir=save_dir, name=f'{model_name}_{i+1} - slice_')
            utils.save_gt(label, save_dir=save_dir, name=f'{model_name}_{i+1} - gt - slice_')
            # utils.save_slice(image=image[..., 8], name=f'Test_{i+1}_slice_8')
            # utils.save_slice(image=image[..., 9], name=f'Test_{i+1}_slice_9')
            # utils.save_slice(image=image[..., 10], name=f'Test_{i+1}_slice_10')
            # utils.save_slice(image=image[..., 11], name=f'Test_{i+1}_slice_11')
            # utils.save_slice(image=image[..., 12], name=f'Test_{i+1}_slice_12')
            # utils.save_gt(label, name=f'Test_{i+1} - gt - slice_')
        '''
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing parameters')
    parser.add_argument('model', type=str, choices=['UNET2D', 'UNET3D', 'UNETR', 'VQGAN', 'SST'], # 'ALL'
                        help='Model to test')
    parser.add_argument('test_set_size', type=int, help='Number of 3D images to test')
    parser.add_argument('-m', '--model_postfix', type=str, default='-1', help='Model postfix to load')
    args = parser.parse_args()
    
    main(args)