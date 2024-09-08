import os
import glob
import logging
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import Dataset

from utilities import image_utils


logger = logging.getLogger(__name__)


class ACDCDataset(Dataset):
    ''' Cardiac 3D image set '''
    
    def __init__(self, dataset_path, slice_size=None, val=False, training=True):
        self.dataset_path = dataset_path
        self.is_training = training
        self.val = val
        self.val_split = 0.1
        self.cardiac_phases = ['ED', 'ES'] # ED - End Diastole, ES - End Systole
        self.num_classes = 4
        self.slice_size = slice_size if slice_size else (128, 128, 21)
        if self.slice_size[-1] < 21:
            logger.warning(f'The ACDC dataset includes 3D images with 21 slices. {self.slice_size} will not use all the data')
        if self.slice_size[0] % 4 != 0 or self.slice_size[1] % 4 != 0:
            logger.warning(f'Suggest slice size to be multiples of 4. Current size: {self.slice_size}')
        
        self.images_train = []
        self.labels_train = []
        self.images_test = []
        self.labels_test = []
        
        # self.image_train_paths = []
        # self.label_train_paths = []
        # self.image_test_paths = []
        # self.label_test_paths = []
        
        self.initialise_data()
        logger.info(f'Loaded ACDC dataset for {"training" if self.is_training else "testing"}')
        
        # Use first 10% of training data for validation
        if val and self.is_training:
            self.images_val = self.images_train[:int(len(self.images_train) * self.val_split)]
            self.labels_val = self.labels_train[:int(len(self.labels_train) * self.val_split)]
            self.images_train = self.images_train[len(self.images_val):]
            self.labels_train = self.labels_train[len(self.labels_val):]


    def __len__(self):
        return len(self.images_train) if self.is_training else len(self.images_test)
        # return len(self.images_train) + len(self.images_test)


    def __getitem__(self, idx):
        # Dimension: NXYZ
        image = self.images_train[idx]

        # Dimension: NXYZ
        label = self.labels_train[idx]
        return image, label
    

    def initialise_data(self):
        # training and test data paths
        paths = {
            'train': os.path.join(self.dataset_path, 'database', 'training'),
            'test': os.path.join(self.dataset_path, 'database', 'testing')
        }
        
        for data_type, data_path in paths.items():
            # no need to load training data when testing
            if data_type == 'train' and not self.is_training:
                continue

            for patient in os.listdir(data_path):
                patient_data_path = os.path.join(data_path, patient)

                # only open patient folders and not MANDATORY_CITATION.md
                if not os.path.isdir(patient_data_path):
                    continue

                frame_nums = self.get_frame_nums(patient_data_path)

                for frame_num in frame_nums:
                    self.load_images_and_labels(patient_data_path, data_type, frame_num)


    def get_frame_nums(self, patient_data_path):
        frame_nums = []
        for line in open(os.path.join(patient_data_path, 'Info.cfg')):
            label, value = line.split(':')
            if label in self.cardiac_phases:
                frame_nums.append(value.rstrip('\n').lstrip(' '))

        for i, frame_num in enumerate(frame_nums):
            if len(frame_num) == 1:
                frame_nums[i] = '0' + frame_nums[i]

        for frame_num in frame_nums:
            assert len(frame_num) == 2, "Error extracting cardiac phase frame number, resulting frame number is not two digits"
        
        return frame_nums


    def load_images_and_labels(self, patient_data_path, data_type, frame_num):
        for file in glob.glob(os.path.join(patient_data_path, f'patient???_frame{frame_num}*.nii.gz')):
            data = nib.load(file).get_fdata()     # Load image data as a NumPy array
            if 'gt' in file:
                if data_type == 'train':
                    self.labels_train.append(data)
                    # self.label_train_paths.append(file)
                if data_type == 'test':
                    self.labels_test.append(data)
                    # self.label_test_paths.append(file)
            else:
                if data_type == 'train':
                    self.images_train.append(data)
                    # self.image_train_paths.append(file)
                if data_type == 'test':
                    self.images_test.append(data)
                    # self.image_test_paths.append(file)


    def get_random_batch(self, batch_size=15, slice_size=None, val=False, training=True, augment=True, index=None):
        '''
        Get a batch of paired images and label maps
        Dimension of images: NCXYZ
        Dimension of labels: NCXYZ
        '''
        images, labels = [], []
        if slice_size is None:
            slice_size = self.slice_size
        
        # get random indices of batch_size length, raise error if batch_size is greater than dataset length
        try:
            if val:
                indices = np.random.choice(len(self.images_val), batch_size, replace=False)
            elif training:
                indices = np.random.choice(len(self.images_train), batch_size, replace=False)
            else:
                indices = np.random.choice(len(self.images_test), batch_size, replace=False)
        except ValueError as e:
            logger.error(f"Batch size {batch_size} is greater than the length of the dataset")
            raise ValueError(f"Batch size {batch_size} is greater than the length of the dataset") from e
        
        # use indices to retrieve images and labels
        if index:
            indices = [index]
        for i in indices:
            if val:
                image, label = self.images_val[i], self.labels_val[i]
            elif training:
                image, label = self.images_train[i], self.labels_train[i]
                if augment:
                    image, label = image_utils.preprocess_data(image, label, img_no=i)
            else:
                image, label = self.images_test[i], self.labels_test[i]
            
            # resize images and labels to slice_size
            image, label = image_utils.resize_3D(image, slice_size), image_utils.resize_3D(label, slice_size)
            image = image_utils.normalise_intensity(image, mode='range')
            # utils.save_image_and_label(image, label, image.shape[-1] // 2, name=f'preprocessed_{i}')
            
            image = image[np.newaxis, ...]
            images.append(image)
            
            # round values in label mask to nearest integer
            label = np.round(label).astype(int)
            label = label[np.newaxis, ...]
            labels.append(label)
            
        try:
            images, labels = np.array(images), np.array(labels)
        except ValueError as e:
            logger.error(f"Failed to convert images and labels to numpy arrays: images: \
                         {len(images), len(images[0]), len(images[0][0]), len(images[0][0][0]), len(images[0][0][0][0])}, \
                         labels: {len(labels), len(labels[0]), len(labels[0][0]), len(labels[0][0][0]), len(labels[0][0][0][0])}")
            raise ValueError(f"Failed to convert images and labels to numpy arrays") from e

        return images, labels


if __name__ == '__main__':
    logging.basicConfig(
        filename=f'/homes/jkl223/Desktop/Individual Project/logs/dataset.log',    # Log file path
        filemode='w',                       # 'a' - append, 'w' - overwrite
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s - %(name)s - %(message)s'
    )
    logger.info('importing utils...')
    
    from utilities import utils

    logger.info('preparing data...')
    dataset_path = '/vol/bitbucket/jkl223/ACDC'
    # dataset = ACDCDataset(dataset_path, val=False, training=False)
    dataset = ACDCDataset(dataset_path, val=True, training=True)
    
    runs = 3
    for iter in range(1, runs + 1):
        logger.info(f'getting random batch... | iter: {iter}')
        batch_size = 50
        # augmented_images, augmented_labels = dataset.get_random_batch(batch_size, val=False, training=False)
        augmented_images, augmented_labels = dataset.get_random_batch(batch_size, val=False, training=True, augment=True)
        # print(augmented_images.shape, augmented_labels.shape)
        for i in range(batch_size):
            augmented_image, augmented_label = augmented_images[i].squeeze(), augmented_labels[i].squeeze()
            # utils.save_image_and_label(augmented_image, augmented_label, name=f'image {i+1} - slice_')

    