from torchvision.transforms import CenterCrop, Compose, Resize, ToTensor

TRANSFORM = Compose([ToTensor(), Resize(256), CenterCrop(128)])
TRANSFORM_CROP_SHIFT = Compose([ToTensor(), Resize(256), CenterCrop(256)])
