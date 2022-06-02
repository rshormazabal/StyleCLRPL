from torchvision import transforms


# TODO: this file needs docstrings
def test_transform(size: int, crop: bool = False) -> transforms.Compose:
    """
    Simple resizing, croping and tensor transform for both style and content images.
    :param size: size for resize (square). [int]
    :param crop: whether to center crop. [bool]
    :return:
    """
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform
