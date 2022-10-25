import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

# note: In case you writing your own dataset, please, make sure that:
#
#       Images 🖼
#           ✅ Images from dataset have the same size, required for packing images to a batch.
#           ✅ Images height and width are divisible by 32. This step is important for segmentation,
#           because almost all models have skip-connections between encoder and decoder and
#           all encoders have 5 downsampling stages (2 ^ 5 = 32).
#           Very likely you will face with error when model will try to concatenate encoder and decoder features if
#           height or width is not divisible by 32.
#           ✅ Images have correct axes order. PyTorch works with CHW order, we read images in HWC [height, width,
#           channels], don`t forget to transpose image.
#       Masks 🔳
#           ✅ Masks have the same sizes as images.
#           ✅ Masks have only 0 - background and 1 - target class values (for binary segmentation).
#           ✅ Even if mask don`t have channels, you need it. Convert each mask from HW to 1HW format for binary
#           segmentation (expand the first dimension).

class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = []
        for im in os.listdir(image_dir):
            if "mask" not in im:
              self.images.append(im)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.jpg"))
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        # label has to be convert from 0-255 to range 0-1
        mask[mask == 255.0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask

