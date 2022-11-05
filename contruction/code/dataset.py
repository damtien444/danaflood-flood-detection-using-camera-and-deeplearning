import json
import os

import cv2
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T

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


class StrFloodDataset(Dataset):
    def __init__(self, image_dir, mask_dir, file_label ,transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform

        with open(file_label) as json_file:
            self.label_dict = json.load(json_file)

        self.images = []
        for im in os.listdir(image_dir):
            if "mask" not in im:
              self.images.append(im)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])

        if ".png" in self.images[index]:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".png", "_mask.jpg"))
        else:
            mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.jpg"))

        image = np.array(Image.open(img_path).convert("RGB"))
        image = canny_preprocess(image)

        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)

        classification = self.label_dict[self.images[index][:-4]] - 1

        # label has to be convert from 0-255 to range 0-1
        mask[mask != 0] = 1

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask, classification

def otsu_thresholding(image):
    otsu_threshold, image_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return otsu_threshold


def auto_canny(image, o_threshold=0.2):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    upper = o_threshold
    lower = upper / 2
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged

def canny_preprocess(img, debug=True):
    _preprocessing = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        lambda x: np.array(x).astype(np.uint8),
        lambda x: auto_canny(x, otsu_thresholding(x)),
        lambda x: cv2.dilate(x, np.ones((3, 3), np.uint8), iterations=1),
        #         lambda x: cv2.morphologyEx(x, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8)),
        lambda x: cv2.bitwise_not(x),
        lambda x: cv2.cvtColor(x,cv2.COLOR_GRAY2RGB),
    ])

#     _postprocessing = T.Compose([
#         T.ToTensor(),
#         T.Resize((512, 512)),
#         T.Normalize(
#            mean=[0.485, 0.456, 0.406],
#            std=[0.229, 0.224, 0.225]
#        ),
#         T.ToPILImage(),
#     ])

    canny_mask = _preprocessing(img)
    applied_mask = cv2.bitwise_and(img, canny_mask)
#     result = _postprocessing(applied_mask)
    return applied_mask