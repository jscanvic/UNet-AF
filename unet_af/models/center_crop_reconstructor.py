import deepinv as dinv
from torchvision.transforms import functional as TF


class CenterCropReconstructor(dinv.models.Reconstructor):
    def __init__(self, base: dinv.models.Reconstructor, crop_size=(128, 128)):
        super().__init__()
        self.base = base
        self.crop_size = crop_size

    def forward(self, y, physics, *args, **kwargs):
        y_cropped = TF.center_crop(y, self.crop_size)
        return self.base(y_cropped, physics, *args, **kwargs)
