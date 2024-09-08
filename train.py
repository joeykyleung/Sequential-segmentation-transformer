import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import traceback

import argparse
from ACDCDataset import ACDCDataset
from utilities.utils import train_test_k_fold, EarlyStopping, BestModel, save_slice, weights_init
from utilities.image_utils import most_freq_pixel_val
from monai.losses import DiceLoss, DiceCELoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.inferers import sliding_window_inference
from monai.networks.utils import one_hot
import optuna


# CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger(__name__)


def main(trial, args):
    Discriminator = None
    if args.model == 'UNET2D':
        from models.UNet2D import UNet2D as model_class, Config
    if args.model == 'UNET3D':
        from models.UNet3D import UNet3D as model_class, Config
    if args.model == 'UNETR':
        from models.UNETR import UNETR as model_class, Config
    if args.model == 'VQGAN':
        from models.VQGAN import VQGAN as model_class, Config
        from models.vqgan.discriminator import Discriminator
        from models.vqgan.lpips import LPIPS
    if args.model == 'SST':
        from models.SST import SST as model_class, Config
        from models.sst.lr_schedule import WarmupLinearLRSchedule
    
    logging.basicConfig( # !!!!
        filename=f'/homes/jkl223/Desktop/Individual Project/logs/train_{Config.name}{"_test" if args.test else ""}.log',    # Log file path
        filemode='w',                       # 'a' - append, 'w' - overwrite
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s - %(name)s - %(message)s'
    )
    logger.info(f'Device: {device}')
    Config.device = device
    
    print(f'Model: {Config.name} | Testing: {args.test} | Cross validation: {args.cross_val} | Epochs: {args.epochs}')
    
    if not args.test:
        # Optuna hyperparameter optimization
        Config.learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        Config.batch_size = trial.suggest_int("batch_size", 4, 8)
        Config.n_layers = trial.suggest_int("n_layers", 12, 36)
        Config.n_heads = trial.suggest_int("n_heads", 4, 16)
        Config.dim = trial.suggest_int("dim", 512, 2048, step=256)
        Config.hidden_dim = trial.suggest_int("hidden_dim", Config.dim*2, Config.dim*8, step=Config.dim)
        Config.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        Config.warmup_epochs = trial.suggest_int("warmup_epochs", 10, 200)
        Config.beta1 = trial.suggest_float("beta1", 0.5, 0.95)
        Config.beta2 = trial.suggest_float("beta2", 0.9, 0.999)
        Config.temperature = trial.suggest_float("temperature", 0.5, 1.5)
        Config.accum_grad = trial.suggest_int("accum_grad", 5, 13)
        
        if Config.dim % Config.n_heads != 0:
            return float('inf')                         # Return a very high loss for invalid configurations
    
    logger.info(f'''
                ###########################################################################
                                    >>>   Starting training   <<<
                Model:                                  {Config.name}
                Cropped image size:                     {Config.slice_size}
                Testing:                                {args.test}
                Cross validation (10 folds):            {args.cross_val}
                Training epochs:                        {args.epochs}
                Batch size:                             {Config.batch_size if not args.test else 1}       
                Loss function:                          {Config.loss_type}
                Optimizer:                              {Config.optimizer}
                ###########################################################################
                ''')
    
    # Model input and output channels
    input_channels = 1
    output_channels = 4
    
    # Segmentation loss
    if Config.loss_type == 'dice':
        criterion = DiceLoss(to_onehot_y=True)#, softmax=True)
    elif Config.loss_type == 'diceCE':
        criterion = DiceCELoss(to_onehot_y=True)
    elif Config.loss_type == 'lpips':
        criterion = LPIPS().eval().to(device=Config.device)
    else:
        criterion = nn.CrossEntropyLoss()
    
    dataset_path = '/vol/bitbucket/jkl223/ACDC'
    if args.cross_val or Config.name in ['VQGAN', 'SST']:
        dataset = ACDCDataset(dataset_path, slice_size=Config.slice_size, val=False, training=True) # obtain all training data
    else:
        dataset = ACDCDataset(dataset_path, slice_size=Config.slice_size, val=True, training=True)
    
    data_len = int(0.25 * len(dataset.images_train))
    indices = np.random.choice(len(dataset.images_train), size=data_len, replace=False)
    dataset.images_train = [dataset.images_train[i] for i in indices]
    dataset.labels_train = [dataset.labels_train[i] for i in indices]
    
    # Create checkpoint folder
    save_dir = '/homes/jkl223/Desktop/Individual Project/models/checkpoints'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    try:
        best_model = train(dataset, model_class, input_channels, output_channels, Config, criterion, args.epochs, 
            save_dir, args.cross_val, args.test, Discriminator, WarmupLinearLRSchedule)
    except RuntimeError as e:
        if 'out of memory' in str(e):
            print(f"CUDA OOM error: {e}")
            # Optionally, clear cache to free up GPU memory
            torch.cuda.empty_cache()
            # Return a high value to indicate failure
            return float('inf')
        else:
            # For other runtime errors, re-raise the exception
            raise e
    except Exception as e:
        print(f"Trial failed with exception: {e}")
        print(traceback.format_exc())
        return float('inf')  # Return a high value to indicate failure
    
    return best_model.best_loss


def train(dataset, model_class, input_channels, output_channels, config, criterion, epochs=10000, 
          save_dir='/', cross_val=False, is_testing=False, Discriminator=None, WarmupLinearLRSchedule=None):
    discriminator_image, discriminator_mask = None, None
    if config.name == 'SST':
        model = model_class(config)
    elif config.name == 'VQGAN':
        model = model_class(config)
        discriminator_image = Discriminator(config, is_mask=False).to(device=config.device)
        discriminator_image.apply(weights_init)
        discriminator_mask = Discriminator(config, is_mask=True).to(device=config.device)
        discriminator_mask.apply(weights_init)
    else:
        model = model_class(input_channels, output_channels)
        discriminator = None
    model = model.to(config.device)
    
    # Optimizer
    optimizer_disc_image, optimizer_disc_mask, optimizer_mask = None, None, None
    if config.name == 'SST':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta1, config.beta2), weight_decay=config.weight_decay)
        lr_schedule = WarmupLinearLRSchedule(optimizer=optimizer,
                                             init_lr=1e-6,
                                             peak_lr=config.learning_rate,
                                             end_lr=0.,
                                             warmup_epochs=epochs//config.warmup_epochs,
                                             epochs=epochs,
                                             current_step=config.start_from_epoch)
    elif config.name == 'VQGAN':
        optimizer = optim.Adam(list(model.encoder.parameters()) +
                            #    list(model.decoder.parameters()) +
                               list(model.decoder_image.parameters()) +
                               list(model.codebook.parameters()) +
                               list(model.quant_conv.parameters()) +
                               list(model.post_quant_conv.parameters()),
                               lr=config.learning_rate, eps=1e-08, betas=(config.beta1, config.beta2))
        optimizer_mask = optim.Adam(list(model.encoder.parameters()) +
                               list(model.decoder_mask.parameters()) +
                               list(model.codebook.parameters()) +
                               list(model.quant_conv.parameters()) +
                               list(model.post_quant_conv.parameters()),
                               lr=config.learning_rate, eps=1e-08, betas=(config.beta1, config.beta2))
        optimizer_disc_image = optim.Adam(discriminator_image.parameters(),
                                    lr=config.learning_rate, eps=1e-08, betas=(config.beta1, config.beta2))
        optimizer_disc_mask = optim.Adam(discriminator_mask.parameters(),
                                    lr=config.learning_rate, eps=1e-08, betas=(config.beta1, config.beta2))
    else:
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)

    # Train the model
    num_iter = epochs
    train_batch_size = config.batch_size #if not is_testing else 1           # !!!
    eval_batch_size = 4 #if not is_testing else 1                            # !!!
    validation_freq = config.validation_freq #if not is_testing else 1       # !!!
    start = time.time()
    
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
    hausdorff = HausdorffDistanceMetric(include_background=True, reduction="mean", get_not_nans=False)
    early_stopping = EarlyStopping(patience=7, name=config.name)
    best_model = BestModel(name=config.name)
    
    if cross_val and config.name != 'VQGAN':
        n_folds = 10                    # 10 to get validation to be 10% of the training data
        fold_indices = train_test_k_fold(n_folds, len(dataset.images_train))
        original_images_train = copy.deepcopy(dataset.images_train)
        original_labels_train = copy.deepcopy(dataset.labels_train)
    else:
        fold_indices = [[-1, -1]]      # dummy value to get one loop
    
    for i, (img_indices, val_indices) in enumerate(fold_indices):
        # reassign dataset with fold indices
        if not (isinstance(img_indices, int) and isinstance(val_indices, int)):
            logger.info(f'----------------------- Fold {i+1} / {len(fold_indices)} -----------------------')
            images_train, labels_train, images_val, labels_val = [], [], [], []
            
            for img in img_indices:
                images_train.append(original_images_train[img])
                labels_train.append(original_labels_train[img])
            for val in val_indices:
                images_val.append(original_images_train[val])
                labels_val.append(original_labels_train[val])
            
            dataset.images_train, dataset.labels_train = images_train, labels_train
            dataset.images_val, dataset.labels_val = images_val, labels_val
        
        if config.name not in ['VQGAN', 'SST']:
            model = model_class(input_channels, output_channels)    # reset model
            model = model.to(config.device)
            optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)     # reset optimizer
        
        early_stopping.__init__(patience=7, name=config.name)   # reset early stopping
        
        # train the model
        for it in range(1, num_iter + 1):
            res = train_iter(model, dataset, config, optimizer, criterion, it, num_iter, i+1, train_batch_size, eval_batch_size, validation_freq, dice_metric, hausdorff,
                             save_dir, early_stopping, best_model, is_testing, lr_schedule, optimizer_mask, discriminator_image, discriminator_mask, optimizer_disc_image, optimizer_disc_mask)
            if res == -1:
                break
    
    logger.info(best_model)
    logger.info('Training took {:.3f}s in total.'.format(time.time() - start))
    
    return best_model


def train_iter(model, dataset, config, optimizer, criterion, it, num_iter, fold, train_batch_size, eval_batch_size, validation_freq, dice_metric, hausdorff, save_dir, 
               early_stopping, best_model, is_testing=False, lr_schedule=None, optimizer_mask=None, discriminator_image=None, discriminator_mask=None, optimizer_disc_image=None, optimizer_disc_mask=None):
    model.train()
        
    # Get a batch of images and labels
    if config.name in ['UNET2D', 'UNET3D', 'VQGAN', 'SST']:
        if config.name == 'VQGAN':
            criterion_mask = DiceCELoss(softmax=True)
        images, labels = dataset.get_random_batch(train_batch_size, val=False, training=True, augment=False)
    else:
        images, labels = dataset.get_random_batch(train_batch_size, val=False, training=True, augment=True)
    images, labels = torch.from_numpy(images), torch.from_numpy(labels)
    images, labels = images.to(config.device, dtype=torch.float32), labels.to(config.device, dtype=torch.float32)
    
    if config.name == 'SST':
        mid_slice = images.shape[-1] // 2
        lr_schedule.step()
        for slc in range(0, images.shape[-1]):
            imgs = images[..., slc]
            lbls = labels[..., slc]
            s_img = copy.deepcopy(imgs)
            _, _, darkest_percent = most_freq_pixel_val(s_img.squeeze().cpu())      # calculates ratio of darkest pixels across batch
            if darkest_percent >= 0.8:
                continue
            
            logits, target = model(imgs, lbls)                                          # [1, 257, 1026], [1, 257]
            logits, target = logits.reshape(-1, logits.size(-1)), target.reshape(-1)    # [257, 1026], [257]
            loss = criterion(logits, target)
            loss.backward()
            logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} | loss for slice {slc+1}: {loss:.8f}')
            
            
            if (it % validation_freq == 0 or it == num_iter) and slc == mid_slice:              # save middle slice for validation
                original_image = imgs[0]
                original_label = lbls[0]
                #dice
                #hausdorff
        best_model(loss, model, fold, it, postfix=f'{"_test" if is_testing else ""}')
        optimizer.step()
        optimizer.zero_grad()
                
    elif config.name == 'VQGAN':
        # images = images.repeat(1, config.image_channels, 1, 1, 1)
        # labels = one_hot(labels, num_classes=4)
        mid_slice = images.shape[-1] // 2
        train_disc = it % config.disc_slow_down == 0
        for slc in range(0, images.shape[-1]):
            imgs = images[..., slc]
            lbls = labels[..., slc]
            s_img = copy.deepcopy(imgs)
            _, _, darkest_percent = most_freq_pixel_val(s_img.squeeze().cpu())      # calculates ratio of darkest pixels across batch
            if darkest_percent >= 0.8:
                continue
            
            # pass through for images
            decoded_images, _, q_loss_img = model(imgs, is_mask=False)
            
            disc_real_img = discriminator_image(imgs)
            disc_fake_img = discriminator_image(decoded_images)
            # disc_factor_img = model.adopt_weight(config.disc_factor, it * slc, threshold=config.disc_start) # to delay start of discriminator loss propagation
            disc_factor_img = 1. if it >= num_iter // config.disc_start else 0.

            perceptual_loss_img = criterion(imgs, decoded_images)
            rec_loss_img = torch.abs(imgs - decoded_images)                             # L1 reconstruction loss
            nll_loss_img = config.perceptual_loss_factor * perceptual_loss_img + config.l2_loss_factor * rec_loss_img
            nll_loss_img = nll_loss_img.mean()
            g_loss_img = -torch.mean(disc_fake_img)                         # generator loss (hinge loss)
            
            λ_img = model.calculate_lambda(nll_loss_img, g_loss_img, is_mask=False)
            loss_vq_img = nll_loss_img + q_loss_img + disc_factor_img * λ_img * g_loss_img
            
            d_loss_real_img = torch.mean(F.relu(1. - disc_real_img))        # discriminator loss (hinge loss) for real images
            d_loss_fake_img = torch.mean(F.relu(1. + disc_fake_img))        # discriminator loss (hinge loss) for fake images
            loss_gan_img = disc_factor_img * .5 * (d_loss_real_img + d_loss_fake_img)
            
            optimizer.zero_grad()
            loss_vq_img.backward(retain_graph=True)
            
            optimizer_disc_image.zero_grad()
            if train_disc:
                loss_gan_img.backward()
            
            optimizer.step()
            if train_disc:
                optimizer_disc_image.step()
            # print("Gradients for image discriminator:", [p.grad for p in discriminator_image.parameters() if p.grad is not None])
            # print("Gradients for generator:", [p.grad for p in model.parameters() if p.grad is not None])
            
            # pass through for labels
            decoded_labels, _, q_loss_mask = model(lbls, is_mask=True)
            
            lbls = one_hot(lbls, num_classes=4)
            loss_lbl = criterion_mask(decoded_labels, lbls)                     # decoded labels come out with 4 channels from VQGAN model
            # decoded_labels = torch.argmax(decoded_labels, dim=1, keepdim=True).type(torch.float32)
            
            disc_real_lbl = discriminator_mask(lbls)
            disc_fake_lbl = discriminator_mask(decoded_labels)
            # disc_factor_lbl = model.adopt_weight(config.disc_factor, it * slc, threshold=config.disc_start)
            disc_factor_lbl = 1. if it >= num_iter // config.disc_start else 0.
            
            # perceptual_loss_lbl = criterion(lbls, decoded_labels)
            rec_loss_lbl = torch.abs(lbls - decoded_labels)             # L1 reconstruction loss for labels
            nll_loss_lbl = config.perceptual_loss_factor * loss_lbl + config.l2_loss_factor * rec_loss_lbl
            nll_loss_lbl = nll_loss_lbl.mean()
            g_loss_lbl = -torch.mean(disc_fake_lbl)                     # generator loss (hinge loss) for labels

            λ_lbl = model.calculate_lambda(nll_loss_lbl, g_loss_lbl, is_mask=True)
            # λ_lbl = model.calculate_lambda(loss_lbl, g_loss_lbl, is_mask=True)
            loss_vq_lbl = nll_loss_lbl + q_loss_mask + disc_factor_lbl * λ_lbl * g_loss_lbl
            # loss_vq_lbl = loss_lbl + q_loss_mask + disc_factor_lbl * λ_lbl * g_loss_lbl
            
            d_loss_real_lbl = torch.mean(F.relu(1. - disc_real_lbl))    # discriminator loss (hinge loss) for real labels
            d_loss_fake_lbl = torch.mean(F.relu(1. + disc_fake_lbl))    # discriminator loss (hinge loss) for fake labels
            loss_gan_lbl = disc_factor_lbl * .5 * (d_loss_real_lbl + d_loss_fake_lbl)
            
            optimizer_mask.zero_grad()
            loss_vq_lbl.backward(retain_graph=True)
            
            optimizer_disc_mask.zero_grad()
            if train_disc:
                loss_gan_lbl.backward()
            
            optimizer_mask.step()
            if train_disc:
                optimizer_disc_mask.step()
            
            if (it % validation_freq == 0 or it == num_iter) and slc == mid_slice:              # save middle slice for validation
                original_image = imgs[0].cpu().detach()
                decoded_image = decoded_images[0].cpu().detach()
                original_label = lbls[0].cpu().detach()
                decoded_label = decoded_labels[0].cpu().detach()
                
                original_image, decoded_image = original_image.unsqueeze(0), decoded_image.unsqueeze(0) #original_image[np.newaxis, ...], decoded_image[np.newaxis, ...]
                original_label, decoded_label = original_label.unsqueeze(0), decoded_label.unsqueeze(0) #original_label[np.newaxis, ...], decoded_label[np.newaxis, ...]
                # print(f'original_image: {original_image.shape}, decoded_image: {decoded_image.shape}, original_label: {original_label.shape}, decoded_label: {decoded_label.shape}')
                
                dice_image = dice_metric(y_pred=decoded_image, y=original_image)
                dice_label = dice_metric(y_pred=decoded_label, y=original_label)
                hausdorff_image = hausdorff(y_pred=decoded_image, y=original_image)
                hausdorff_label = hausdorff(y_pred=decoded_label, y=original_label)
                
                original_image, decoded_image = original_image.squeeze().numpy(), decoded_image.squeeze().numpy()
                original_label, decoded_label = np.argmax(original_label.squeeze().numpy(), axis=0), np.argmax(decoded_label.squeeze().numpy(), axis=0)
                
                print(f'decoded_image: {np.min(decoded_image)} - {np.max(decoded_image)}, decoded_label: {np.min(decoded_label)} - {np.max(decoded_label)}')
            # print("Gradients for mask discriminator:", [p.grad for p in discriminator_mask.parameters() if p.grad is not None])
            # print("Gradients for generator:", [p.grad for p in model.parameters() if p.grad is not None])
            logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} | Slice {slc+1}, Image VQ loss: {loss_vq_img.cpu().detach().numpy().item():.3f}, Image GAN loss: {loss_gan_img.cpu().detach().numpy().item():.3f}')
            logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} |                Image VQ: nll - {nll_loss_img:.3f}, q_loss - {q_loss_img:.3f}, fac - {disc_factor_img:.3f}, λ - {λ_img:.3f}, g_loss - {g_loss_img:.3f}')
            logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} |                Image GAN: fac - {disc_factor_img:.3f}, d_loss_real - {d_loss_real_img:.3f}, d_loss_fake - {d_loss_fake_img:.3f}')
            logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} | Slice {slc+1}, Label VQ loss: {loss_vq_lbl.cpu().detach().numpy().item():.3f}, Label GAN loss: {loss_gan_lbl.cpu().detach().numpy().item():.3f}')
            logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} |                Label VQ: nll - {nll_loss_lbl:.3f}, q_loss - {q_loss_mask:.3f}, fac - {disc_factor_lbl:.3f}, λ - {λ_lbl:.3f}, g_loss - {g_loss_lbl:.3f}')
            logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} |                Label GAN: fac - {disc_factor_lbl:.3f}, d_loss_real - {d_loss_real_lbl:.3f}, d_loss_fake - {d_loss_fake_lbl:.3f}')
    
    elif config.name == 'UNET2D':                           # Convert 3D imgaes to 2D if training UNET2D
        for slc in range(0, images.shape[-1]):
            s_img = copy.deepcopy(images[..., slc])
            # save_slice(image=s_img.squeeze().cpu(), name=f'2d_slice_{slc+1}')
            _, _, darkest_percent = most_freq_pixel_val(s_img.squeeze().cpu())      # calculates ratio of darkest pixels across batch
            if darkest_percent >= 0.8:                                              # do not train on empty slices
                continue
            
            logits = model(images[..., slc])
            optimizer.zero_grad()
            loss = criterion(logits, labels[..., slc])
            loss.backward()
            optimizer.step()
            logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} | loss for slice {slc+1}: {loss:.8f}')
    else:
        logits = model(images)
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} | loss: {loss:.8f}')
        
    # Validation
    if it % validation_freq == 0 or it == num_iter:
        if config.name == 'SST':
            with torch.no_grad():
                # log, sampled_imgs = model.log_images(original_image.unsqueeze(0))
                log = model.log_images(original_image.unsqueeze(0))
                save_slice(image=log["input"].cpu().detach().numpy()[0].transpose(1, 2, 0), 
                           label=original_label.cpu().detach().numpy().squeeze(), name=f'input_{it}')
                # sampled_imgs.add(1).mul(0.5)
                # save_slice(label=sampled_imgs.cpu().detach().numpy()[0].transpose(1, 2, 0), name=f'result_{it}')
                    # image = log["input"].cpu().detach().numpy()[0].transpose(1, 2, 0)
                    # print(f'\tlog["input"]: {image.shape}')
                save_slice(label=log["output"].cpu().detach().numpy()[0].transpose(1, 2, 0), name=f'output_{it}')
                # save_slice(label=log["half_sample"].cpu().detach().numpy()[0].transpose(1, 2, 0), name=f'half_sample_{it}')
                # save_slice(label=log["new_sample"].cpu().detach().numpy()[0].transpose(1, 2, 0), name=f'new_sample_{it}')
                logger.info(f'Saved images for epoch {it}... dice... hausdorff...')
            # torch.save(model.state_dict(), f'{save_dir}/{config.name}_{fold}_{it}.pt')               
        elif config.name == 'VQGAN':
            with torch.no_grad():
                # original_image, original_label = np.argmax(original_image, axis=0), np.argmax(original_label, axis=0)
                # decoded_image, decoded_label = np.argmax(decoded_image, axis=0), np.argmax(decoded_label, axis=0)
                decoded_image, decoded_label = (decoded_image + 1) * 0.5, (decoded_label + 1) * 0.5
                save_slice(image=original_image, label=original_label, name=f'original_{it}')
                save_slice(image=decoded_image, label=decoded_label, name=f'decoded_{it}')
                logger.info(f'Saved images for epoch {it} | dice image: {dice_image}, hausdorff image: {hausdorff_image}')
                logger.info(f'                              dice label: {dice_label}, hausdorff label: {hausdorff_label}')
            torch.save(model.state_dict(), f'{save_dir}/{config.name}_{fold}_{it}.pt')
        else:
            res = validate_iter(model, dataset, config, it, num_iter, fold, eval_batch_size, 
                                loss, dice_metric, early_stopping, best_model, is_testing)
            if res == -1:
                return -1
    
    return 0


def validate_iter(model, dataset, config, it, num_iter, fold, eval_batch_size, training_loss, 
                  dice_metric, early_stopping, best_model, is_testing=False):
    model.eval()
    test_images, test_labels = dataset.get_random_batch(eval_batch_size, val=True, training=True)
    test_images, test_labels = torch.from_numpy(test_images), torch.from_numpy(test_labels)
    test_images, test_labels = test_images.to(config.device, dtype=torch.float32), test_labels.to(config.device, dtype=torch.float32)

    with torch.no_grad():
        # Convert 3D imgaes to 2D if testing UNET2D
        if config.name == 'UNET2D':
            masks = torch.zeros_like(test_labels)
            for slc in range(0, test_images.shape[-1]):
                test_logits = model(test_images[..., slc])
                p = torch.softmax(test_logits, dim=1)
                output = torch.argmax(p, dim=1)
                output = output[:, np.newaxis, ...]
                masks[..., slc] = output
        else:
            test_logits = model(test_images)
            p = torch.softmax(test_logits, dim=1)
            output = torch.argmax(p, dim=1)
            masks = output[:, np.newaxis, ...]
        
        num_classes = 4
        y_pred = one_hot(masks, num_classes=num_classes)
        y = one_hot(test_labels, num_classes=num_classes)
        
        dice_metric(y_pred=y_pred, y=y)

    test_loss = 1 - dice_metric.aggregate().item()
    dice_metric.reset()
    logger.info(f'Fold: {fold} | Epoch: {it} / {num_iter} | training loss: {training_loss:.8f}, test dice loss: {test_loss:.8f}')
    
    # check for early stopping and save model if validation performance improved
    early_stopping(test_loss, model, fold, it, logger, postfix=f'{"_test" if is_testing else ""}', best_loss_overall=best_model.best_loss)
    best_model(test_loss, model, fold, it, postfix=f'{"_test" if is_testing else ""}')             # !!!!
    
    if early_stopping.early_stop:
        return -1
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('-t', '--test', action='store_true', help='Use test params (1 batch, validation every epoch)')
    parser.add_argument('-x', '--cross_val', action='store_true', help='Use cross validation (10 folds)')
    parser.add_argument('model', type=str, choices=['UNET2D', 'UNET3D', 'UNETR', 'VQGAN', 'SST'],
                        help='Model to train')
    parser.add_argument('epochs', type=int, help='Number of epochs to train for')
    args = parser.parse_args()
    
    study = optuna.create_study(direction="minimize")
    if args.test:
        # main(None, args)
        study.optimize(lambda trial: main(trial, args), n_trials=1)
    else:
        study.optimize(lambda trial: main(trial, args), n_trials=50)
        best_trial = study.best_trial
        
        print("Number of finished trials: ", len(study.trials))
        print(f"  Best Loss: {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")