"""Utility functions for Shared Interest."""
import cv2
import numpy as np
import skimage


def flatten(batch):
    """
    Flattens saliency by summing the channel dimension.

    Args:
    batch: 4D numpy array (batch, channels, height, width).

    Returns: 3D numpy array (batch, height, width) with the channel dimension
        summed.
    """
    return np.sum(batch, axis=1)


def normalize_0to1(batch):
    """
    Normalize a batch such that every value is in the range 0 to 1.

    Args:
    batch: a batch first numpy array to be normalized.

    Returns: A numpy array of the same size as batch, where each item in the
    batch has 0 <= value <= 1.
    """
    axis = tuple(range(1, len(batch.shape)))
    minimum = np.min(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    maximum = np.max(batch, axis=axis).reshape((-1,) + (1,) * len(axis))
    normalized_batch = (batch - minimum) / (maximum - minimum)
    return normalized_batch


def binarize_percentile(batch, percentile):
    """
    Creates binary mask by thresholding at percentile.

    Args:
    batch: 4D numpy array (batch, height, width).
    percentile: float in range 0 to 1. Values above the percentile value are 
        set to 1. Values below the percentile value are set to 0.

    Returns: A 4D numpy array with dtype uint8 with all values set to 0 or 1.
    """
    batch_size = batch.shape[0]
    batch_normalized = normalize_0to1(batch)
    percentile = np.percentile(batch_normalized, percentile * 100, axis=(1, 2)).reshape(batch_size, 1, 1)
    binary_mask = (batch_normalized >= percentile).astype('uint8')
    return binary_mask


def binarize_std(batch, num_std=1):
    """
    Creates binary mask by thresholding at num_std standard deviations above
    the mean.

    Args:
    batch: 3D numpy array (batch, height, width).
    num_std: int in range 0 to 3. Values above the (mean + num_std * std) value
        are set to 1. Values below are set to 0.

    Returns: A 3D numpy array with dtype uint8 with all values set to 0 or 1.
    """
    batch_size = batch.shape[0]
    batch_normalized = normalize_0to1(batch)
    mean = np.mean(batch_normalized, axis=(1, 2)).reshape(batch_size, 1, 1)
    std = np.std(batch_normalized, axis=(1, 2)).reshape(batch_size, 1, 1)
    threshold = mean + num_std * std
    binary_mask = (batch_normalized >= threshold).astype('uint8')
    return binary_mask


def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.
    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")

    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def component_analysis(array_mask):
    
    labeled_image = skimage.measure.label(array_mask[0,:,:]>0,connectivity=2,return_num=True) # labeled_image[0].shape (224, 224, 3)  # https://datacarpentry.org/image-processing/09-connected-components/
    
    num_of_components = labeled_image[1]
    object_features = skimage.measure.regionprops(labeled_image[0])
    area_of_components = [int(objf["area"]) for objf in object_features]
    
    area_of_components_percentage = np.array(area_of_components)/(array_mask.shape[1]*array_mask.shape[2])

    return num_of_components,area_of_components, area_of_components_percentage.tolist()


def save_grayscale(filename, array):
    array = (array - array.min()) / (array.max() - array.min())
    array = array.squeeze(0).squeeze(0) * 255
    cv2.imwrite(filename, array)
