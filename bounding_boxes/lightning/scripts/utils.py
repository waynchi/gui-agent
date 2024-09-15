import cv2
import numpy as np

def resize_and_pad(img, output_size=(1024, 1024)):
    # Read the image
    h, w, _ = img.shape

    # Calculate the scale factor and new dimensions
    scale = output_size[0] / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img_rescaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Calculate padding
    top_pad = (output_size[1] - new_h) // 2
    bottom_pad = output_size[1] - new_h - top_pad
    left_pad = (output_size[0] - new_w) // 2
    right_pad = output_size[0] - new_w - left_pad

    # Apply padding
    img_padded = cv2.copyMakeBorder(img_rescaled, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return img_padded

def reverse_resize_and_pad(bboxes, orig_height, orig_width, output_size=(1024, 1024)):
    """
    Adjust bounding box coordinates to match the original image size before resizing and padding.
    
    Args:
        bboxes (np.array): Bounding boxes in the format [[x1, y1, x2, y2], ...] adjusted to the resized and padded image.
        orig_height (int): Original height of the image before resizing and padding.
        orig_width (int): Original width of the image before resizing and padding.
        output_size (tuple): The size to which the image was resized and padded.
    
    Returns:
        np.array: Adjusted bounding boxes in the original image's scale.
    """
    scale = max(orig_height, orig_width) / output_size[0]
    
    # Calculate the padding added during resizing
    if orig_width > orig_height:
        new_height = int(orig_height * (output_size[0] / orig_width))
        pad_vertical = (output_size[1] - new_height) // 2
        pad_horizontal = 0
    else:
        new_width = int(orig_width * (output_size[1] / orig_height))
        pad_horizontal = (output_size[0] - new_width) // 2
        pad_vertical = 0

    # Reverse padding
    bboxes[:, [0, 2]] -= pad_horizontal
    bboxes[:, [1, 3]] -= pad_vertical

    # Reverse scaling
    bboxes /= scale
    
    return bboxes
