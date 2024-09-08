import os
import torch
import numpy as np
import torch.nn as nn
from numpy.random import default_rng
import matplotlib.pyplot as plt
from matplotlib import colors


save_dir = '/homes/jkl223/Desktop/Individual Project/results/test'
os.makedirs(save_dir, exist_ok=True)


def save_image_and_label(image, label, slc=None, save_dir=save_dir, name='slice_', figsize=(4, 4)):
    shortest_axis = np.argmin(image.shape)
    slc_range = image.shape[shortest_axis] if slc is None else 1    # if slice is specified, only save that slice
    
    for i in range(slc_range):
        fig, ax = plt.subplots(figsize=figsize)
        if slc:
            ax.imshow(image[..., slc], cmap='gray', alpha=0.5)
            ax.imshow(label[..., slc], cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']), alpha=0.3)
        else:
            if shortest_axis == 0:
                ax.imshow(image[i, ...], cmap='gray', alpha=0.5)
                ax.imshow(label[i, ...], cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']), alpha=0.3)
            elif shortest_axis == 1:
                ax.imshow(image[:, i, ...], cmap='gray', alpha=0.5)
                ax.imshow(label[:, i, ...], cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']), alpha=0.3)
            else:
                ax.imshow(image[..., i], cmap='gray', alpha=0.5)
                ax.imshow(label[..., i], cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']), alpha=0.3)
        
        ax.axis('off')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{name}{i+1}.png') if slc is None \
                                                               else os.path.join(save_dir, f'{name}.png')

        plt.savefig(save_path)
        plt.close()  # Close the figure to release memory


def save_slice(image=None, label=None, save_dir=save_dir, name='2D_slice_', figsize=(4, 4)):
    if image is None and label is None:
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    if image is not None and label is not None:
        ax.imshow(image, cmap='gray', alpha=0.5)
        ax.imshow(label, cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']), alpha=0.3)
    elif image is not None:
        ax.imshow(image, cmap='gray')
    elif label is not None:
        # Plot a black background
        ax.imshow(np.zeros_like(label), cmap='gray', vmin=0, vmax=1)
        cmap_colors = ['black'] + plt.cm.viridis(np.linspace(0, 1, len(np.unique(label)) - 1)).tolist()
        custom_cmap = colors.ListedColormap(cmap_colors)
        ax.imshow(label, cmap=custom_cmap) #colors.ListedColormap(['black', 'green', 'blue', 'red']))
    
    ax.axis('off')
    plt.tight_layout()

    save_path = os.path.join(save_dir, f'{name}.png')

    plt.savefig(save_path)
    plt.close()


def save_gt(gt, save_dir=save_dir, name='slice_', figsize=(4, 4)):
    shortest_axis = np.argmin(gt.shape)
    for i in range(gt.shape[shortest_axis]):
        fig, ax = plt.subplots(figsize=figsize)
        if shortest_axis == 0:
            ax.imshow(gt[i, ...], cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']))
        elif shortest_axis == 1:
            ax.imshow(gt[:, i, ...], cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']))
        else:
            ax.imshow(gt[..., i], cmap=colors.ListedColormap(['black', 'green', 'blue', 'red']))
        
        ax.axis('off')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f'{name}{i+1}.png')

        plt.savefig(save_path)
        plt.close()

    
def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.

    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a 
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices


def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class EarlyStopping:
    def __init__(self, patience=5, name='model'):
        self.patience = patience
        self.name = name
        self.best_loss = float('inf')
        self.best_loss_fold = -1
        self.best_loss_epoch = -1
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model, fold, it, logger, postfix='', best_loss_overall=0.2):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_loss_fold = fold
            self.best_loss_epoch = it
            self.counter = 0
            
            # Save the model checkpoint if loss is < best out of all folds     !!!!!
            if self.best_loss < min(best_loss_overall, 0.15):
                self.best_model = model.state_dict()
                torch.save(self.best_model, f'/homes/jkl223/Desktop/Individual Project/models/checkpoints/{self.name}_{self.best_loss_fold}_{self.best_loss_epoch}{postfix}.pt')
                logger.info(f'Saved best model so far ({self.name}_{self.best_loss_fold}_{self.best_loss_epoch}{postfix}.pt) with dice loss: {self.best_loss}')
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info(f'Early stopping: loss did not improve for {self.patience} consecutive epochs.')
                if self.best_loss_fold == -1 and self.best_loss_epoch == -1:
                    logger.info(f'Best model in a previous fold. No model was saved')
                else:
                    logger.info(f'Best model for fold {fold} was {self.name}_{self.best_loss_fold}_{self.best_loss_epoch}{postfix}.pt with dice loss: {self.best_loss}')
    def __str__(self):
        return f'early_stopping - {self.early_stop} | count: {self.counter}, p: {self.patience}, best_loss: {self.best_loss}, best_loss_fold: {self.best_loss_fold}, best_loss_epoch: {self.best_loss_epoch}'


class BestModel:
    def __init__(self, name='model'):
        self.name = name
        self.best_loss = float('inf')
        self.best_loss_fold = -1
        self.best_loss_epoch = -1

    def __call__(self, val_loss, model, fold, it, postfix=''):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_loss_fold = fold
            self.best_loss_epoch = it
            torch.save(model.state_dict(), f'/homes/jkl223/Desktop/Individual Project/models/checkpoints/{self.name}_best{postfix}.pt') #{self.best_loss_fold}_{self.best_loss_epoch}{postfix}.pt')
            
    def __str__(self):
        return f'------ Best model overall was {self.name}_{self.best_loss_fold}_{self.best_loss_epoch}.pt with dice loss: {self.best_loss} ------'
