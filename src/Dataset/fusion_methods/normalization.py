

import numpy as np

def normalize(image):
    fused_image = image
    # Ensure range 0-255 and uint8 encoding
    fused_image = (fused_image - fused_image.min()) / (fused_image.max() - fused_image.min())
    fused_image = (fused_image * 255).astype(np.uint8)
    fused_image = (fused_image * 255).astype(np.uint8)
    return fused_image