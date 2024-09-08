import torch
import numpy as np
from scipy.ndimage import rotate, zoom, gaussian_filter

from utilities import utils
from monai.transforms import (
    Compose, EnsureChannelFirstD, ScaleIntensityRangeD, NormalizeIntensityD, RandRotateD, 
    RandAffineD, RandFlipD, RandGaussianNoiseD, RandGaussianSmoothD, ToTensorD
)


def normalise_intensity(image, mode='z-score', thres_roi=1.0):
    if mode not in ['z-score', 'range']:
        raise ValueError(f"Normalisation mode must be 'z-score' or 'range', got: {mode}")
    
    if mode == 'z-score':
        ''' Normalise the image intensity by the mean and standard deviation '''
        # ROI defines the image foreground (more than 1.0 percentile)
        val_l = np.percentile(image, thres_roi)
        roi = (image >= val_l)
        mu, sigma = np.mean(image[roi]), np.std(image[roi])
        eps = 1e-6
        normalised_image = (image - mu) / (sigma + eps)
        
    elif mode == 'range':
        ''' Normalise the image to [0, 1] '''
        max_val, min_val = np.max(image), np.min(image)
        normalised_image = (image - min_val) / (max_val - min_val)
    
    return normalised_image


def resize_3D(data, output_size, percentile=5):
    ''' Resize 3D image by cropping or padding each 2D slice '''
    if data.ndim != 3:
        raise ValueError(f"Input data must be 3D, got shape: {data.shape}")
    
    if data.shape[0] == output_size[0] and data.shape[1] == output_size[1] and data.shape[2] == output_size[2]:
        return data
    
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    
    _, _, input_slices = data.shape
    output_height, output_width, output_slices = output_size
    
    # Determine start and end slices for input to fit in the centre of output
    start_slice = max((output_slices - input_slices) // 2, 0)
    end_slice = start_slice + min(input_slices, output_slices)
    
    flattened_image = data.flatten()
    pad_val = np.min(flattened_image)
    
    resized_data = np.full((output_height, output_width, output_slices), pad_val, dtype=data.dtype)
    # resized_data = np.zeros((output_height, output_width, output_slices), dtype=data.dtype)
    
    for zz in range(output_slices):
        if start_slice <= zz < end_slice:
            resized_data[:, :, zz] = crop_or_pad_2D_slice(data[:, :, zz - start_slice], output_height, output_width, pad_val=pad_val)
    
    return resized_data


def crop_or_pad_2D_slice(slice, nx, ny, pad_val=0):
    x, y = slice.shape   # number of rows and columns

    x_s = (x - nx) // 2  # Start index for cropping (rows)
    y_s = (y - ny) // 2  # Start index for cropping (columns)
    x_c = (nx - x) // 2  # Amount of padding (rows)
    y_c = (ny - y) // 2  # Amount of padding (columns)

    if x > nx and y > ny:
        # If slice is larger than target
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.full((nx, ny), pad_val, dtype=slice.dtype)
        # slice_cropped = np.zeros((nx, ny), dtype=slice.dtype)

        if x <= nx and y > ny:
            # If slice is shorter in rows but longer in columns, pad rows
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            # If slice is shorter in columns but longer in rows, pad columns
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            # If slice is smaller than target, pad both rows and columns
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def random_flip(image, label, p=0.5):
    if np.random.rand() < p:
        axis = np.random.randint(0, 3)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label, min=-30, max=30, p=0.2):
    if np.random.rand() < p:
        # flattened_image, flattened_label = image.flatten(), label.flatten()
        # min_val_image, min_val_label = np.min(flattened_image), np.min(flattened_label)
        
        for axes in [(0, 1), (0, 2), (1, 2)]:
            angle = np.random.uniform(min, max)
            image = rotate(image, angle, axes=axes, order=3, reshape=False)#, mode='constant', cval=min_val_image)
            label = rotate(label, angle, axes=axes, order=0, reshape=False)#, mode='constant', cval=min_val_label)
    return image, label


def random_scale(image, label, min=0.7, max=1.3, p=0.2): # max scale 1.3 instead of 1.4 from nnUNet so gt will fit in sample
    if np.random.rand() < p:
        # flattened_image, flattened_label = image.flatten(), label.flatten()
        # min_val_image, min_val_label = np.min(flattened_image), np.min(flattened_label)
        
        scale = np.random.uniform(min, max)
        original_shape = image.shape
        
        # Scale the entire 3D image
        image = zoom(image, zoom=scale, order=3)#, mode='constant', cval=min_val_image)
        label = zoom(label, zoom=scale, order=0)#, mode='constant', cval=min_val_label)
        
        image = resize_3D(image, original_shape)
        label = resize_3D(label, original_shape)
        
    return image, label


def add_gaussian_noise(image, mean=0, p=0.15):
    if np.random.rand() < p:
        std = np.random.uniform(0, 0.1)
        noise = np.random.normal(mean, std, image.shape)
        image = image + noise
    return image


def gaussian_blur(image, p=0.1):
    if np.random.rand() < p:
        sigma = np.random.uniform(0.5, 1.5)
        image = gaussian_filter(image, sigma=sigma)
    return image


def most_freq_pixel_val(image, percentile=15):
    if isinstance(image, torch.Tensor):
        image = image.numpy()
    
    # get middle 3 slices
    if image.ndim == 3:
        middle_slice = image.shape[-1] // 2
        image = image[..., middle_slice - 1 : middle_slice + 2]    

    flattened_image = image.flatten()
    if flattened_image.size == 0:
        raise ValueError(f"Empty image, no pixels found. Image shape: {image.shape}")
    
    # Count the number of dark pixels (min value of image)
    flattened_image = image.flatten()
    min_val = np.min(flattened_image)
    darkest_percentile = np.percentile(flattened_image, percentile)
    dark_pixel_count = np.sum(flattened_image <= darkest_percentile)
    total_pixel_count = flattened_image.size
    
    # Calculate the proportion of black pixels
    dark_pixel_ratio = dark_pixel_count / total_pixel_count
    
    return min_val, darkest_percentile, dark_pixel_ratio #most_freq_val


def preprocess_data(image, label, max_iter=10, num_iter=1, img_no=0):
        '''
        image: rot -> scale -> flip -> noise -> blur -> crop or pad -> normalise
        label: rot -> scale -> flip                  -> crop or pad
        '''
        
        original_image = image.copy()
        original_label = label.copy()
        
        image = normalise_intensity(image, mode='range')
        
        image, label = random_rotate(image, label)
        image, label = random_scale(image, label)
        image, label = random_flip(image, label)
        
        image = normalise_intensity(image, mode='range')
        
        # utils.save_image_and_label(image, label, image.shape[-1] // 2, name=f'preprocessed_{img_no}')
        min_val, darkest_val, darkest_percent = most_freq_pixel_val(image)#[..., image.shape[-1] // 2])
        sums = np.array([image[..., i].sum() for i in range(image.shape[-1])])
        range_z = sums.max() - sums.min()
        # print(f'preprocessed image {img_no}, %darkest pixels: {darkest_percent:.2f}, val: {darkest_val:.3f} (min {min_val} - {darkest_val == min_val}), max_diff: {range_z}')
        
        # preprocess original data again if resulting image is too distorted (check middle 3 slices)
        if num_iter <= max_iter and (darkest_percent >= 0.4 or range_z <= 300):
            # print(f'\timage {img_no} is too distorted, %darkest pixels: {darkest_percent}, val: {darkest_val}, range_z: {range_z}')
            return preprocess_data(original_image, original_label, max_iter, num_iter + 1, img_no)
        
        image = add_gaussian_noise(image)
        image = gaussian_blur(image)
        
        return image, label
        
        '''
        transforms = Compose([
            EnsureChannelFirstD(keys=['image', 'label'], strict_check=False, channel_dim=0),
            ScaleIntensityRangeD(keys=['image'], a_min=-500, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
            NormalizeIntensityD(keys=['image'], nonzero=True, channel_wise=True),  # Normalize around mean and std
            RandRotateD(keys=['image', 'label'], range_x=15, range_y=15, range_z=15, prob=0.2),
            RandAffineD(keys=['image', 'label'], prob=0.2, scale_range=(0.8, 1.2), mode=("bilinear", "nearest")),
            RandFlipD(keys=['image', 'label'], prob=0.5, spatial_axis=[0, 1, 2]),
            RandGaussianNoiseD(keys=['image'], std=0.1, prob=0.15),
            RandGaussianSmoothD(keys=['image'], sigma_x=(0.5, 1.5), prob=0.1),
            # ToTensorD(keys=['image', 'label'], dtype=torch.float32),
        ])
        
        data = {'image': image[np.newaxis, ...], 'label': label[np.newaxis, ...]}
        data = transforms(data)
        
        return data['image'].squeeze(), data['label'].squeeze()
        '''

"""
class AugmentationPipeline:
    ''' Image augmentations (rotation and scaling, gaussian noise, gaussian blur, mirroring) per nnUNet configuration '''
    def __init__(self, slize_size):
        self.image_and_label_transforms = []
        self.image_transforms = []
        
        # Z-normalization
        self.image_and_label_transforms.append(tio.ZNormalization())

        # Define padding to ensure no cropping occurs during transformations
        max_rotation_degrees = 30
        max_scaling_factor = 1.4
        max_padding = self.calculate_dynamic_padding(max_rotation_degrees, max_scaling_factor)

        # Image and label transformations (rotation, scaling, mirroring)
        self.image_and_label_transforms.append(tio.transforms.Pad(max_padding))
        self.image_and_label_transforms.append(
            tio.transforms.RandomAffine(
                degrees=(max_rotation_degrees, max_rotation_degrees, max_rotation_degrees), 
                image_interpolation='linear', p=1))#0.2))
        self.image_and_label_transforms.append(
            tio.transforms.RandomAffine(
                scales=(0.7, max_scaling_factor), 
                image_interpolation='linear', p=1))#0.2))
        self.image_and_label_transforms.append(tio.transforms.RandomFlip(axes=(0, 1, 2), flip_probability=0.5))
        self.image_and_label_transforms.append(tio.CropOrPad(slize_size))
        
        # Image transformations (add noise and blur to images)
        self.image_transforms.append(tio.transforms.RandomNoise(mean=0, std=(0, 0.1), p=0.15))
        self.image_transforms.append(tio.transforms.RandomBlur(std=(0.5, 1.5), p=0.1))
        
        # Compose transforms
        self.image_and_label_transforms = tio.Compose(self.image_and_label_transforms)
        self.image_transforms = tio.Compose(self.image_transforms)
        
        
    def calculate_dynamic_padding(self, max_rotation_degrees, max_scaling_factor):
        # Compute the maximum extent of padding required based on rotation and scaling
        radians = np.radians(max_rotation_degrees)
        max_extent = np.sqrt(2) * (1 - np.cos(radians))
        max_scaling = max_scaling_factor - 1
        # Considering maximum extent for both scaling and rotation
        padding = int(np.ceil(max_extent * 100)) + int(np.ceil(max_scaling * 100))
        return padding


    def apply(self, image, label):
        '''
        Apply the augmentation pipeline to the given image and label.
        
        Parameters:
        image (torch.Tensor or numpy.ndarray): The input 3D image.
        label (torch.Tensor or numpy.ndarray): The corresponding label.
        
        Returns:
        (torch.Tensor, torch.Tensor): The augmented 3D image and label.
        '''
        # Ensure image and label are torch tensors
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if isinstance(label, np.ndarray):
            label = torch.from_numpy(label)
            
        original_image_shape = image.shape  # Save the original shape for cropping later

        # Create a Subject with both image and label
        subject = tio.Subject(
            image=tio.Image(tensor=image, type=tio.INTENSITY),
            label=tio.Image(tensor=label, type=tio.LABEL)
        )

        # Apply the transformations to both image and label
        transformed_subject = self.image_and_label_transforms(subject)

        # Apply the image-only transformations to the image part of the transformed subject
        transformed_subject.image = self.image_transforms(transformed_subject.image)

        # Extract the transformed image and label
        transformed_image = transformed_subject.image.data
        transformed_label = transformed_subject.label.data
        
        # Crop back to the original size if padding was added
        # pad = self.padding_size
        # transformed_image = transformed_image[
        #     :,
        #     pad:-pad,
        #     pad:-pad,
        #     pad:-pad
        # ]
        # transformed_label = transformed_label[
        #     :,
        #     pad:-pad,
        #     pad:-pad,
        #     pad:-pad
        # ]
        
        return transformed_image, transformed_label
"""