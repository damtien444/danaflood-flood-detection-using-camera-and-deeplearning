import albumentations as albu
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

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

def canny_preprocess(img, **args):
    _preprocessing = T.Compose([
        T.ToPILImage(),
        T.Grayscale(),
        lambda x: np.array(x).astype(np.uint8),
        lambda x: auto_canny(x, otsu_thresholding(x)),
        # lambda x: cv2.dilate(x, np.ones((3, 3), np.uint8), iterations=1),
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

    canny_mask = _preprocessing(img.astype(np.uint8))
    # print(img.shape, canny_mask.shape, img.dtype, canny_mask.dtype)
    applied_mask = cv2.bitwise_and(img, canny_mask)
    # result = _postprocessing(applied_mask)
    return applied_mask

def get_training_augmentation(is_preprocessing=False):
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        #
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=5, shift_limit=0.2, p=0.5, border_mode=0),
        #
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        albu.RandomCrop(height=512, width=512, p=0.2),
        albu.Resize(height=512, width=512),
        #
        albu.GaussNoise(p=0.5),
        albu.Perspective(p=0.5),
        #
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        #
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        #
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),

    ]
    if is_preprocessing:
        train_transform.extend([
            albu.Lambda(image=canny_preprocess)
        ])

    train_transform.extend([albu.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0,1.0,1.0], max_pixel_value=255.0), ToTensorV2()])
    # train_transform.extend([ToTensorV2()])
    return albu.Compose(train_transform)

def get_validation_augmentation(is_preprocessing=False):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(512, 512),
        # albu.RandomCrop(height=512, width=512, always_apply=True),
        albu.Resize(height=512, width=512),


    ]
    if is_preprocessing:
        test_transform.extend([
            albu.Lambda(image=canny_preprocess)
        ])

    test_transform.extend([albu.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0,1.0,1.0], max_pixel_value=255.0), ToTensorV2()])
    return albu.Compose(test_transform)

# def to_tensor(x, **kwargs):
#     return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn, is_preprocess):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """
    if is_preprocess:
        _transform = [
            albu.Lambda(image=canny_preprocess),
            ToTensorV2(),
        ]
    else:
        _transform = [
            ToTensorV2(),
        ]
    return albu.Compose(_transform)