from PIL import Image


def resize_to_max_dim(image: Image.Image, max_dim: int = 1024) -> Image.Image:
    """
    Resize an image so that neither width nor height exceeds max_dim,
    while preserving the aspect ratio.
    If the image is already within limits, it is returned unchanged.
    """
    width, height = image.size

    if width <= max_dim and height <= max_dim:
        return image

    # Calculate the scaling factor
    scale = min(max_dim / width, max_dim / height)
    new_width = int(width * scale)
    new_height = int(height * scale)

    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)