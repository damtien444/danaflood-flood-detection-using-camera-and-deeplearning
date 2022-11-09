import albumentations as albu
from albumentations.pytorch import ToTensorV2


def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),
        #
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=5, shift_limit=0.1, p=1, border_mode=0),
        #
        albu.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
        # albu.RandomCrop(height=512, width=512, always_apply=True),
        albu.Resize(height=512, width=512),
        #
        # albu.GaussNoise(p=0.2),
        # albu.Perspective(p=0.5),
        #
        # albu.OneOf(
        #     [
        #         albu.CLAHE(p=1),
        #         albu.RandomBrightness(p=1),
        #         albu.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.Sharpen(p=1),
        #         albu.Blur(blur_limit=3, p=1),
        #         albu.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        #
        # albu.OneOf(
        #     [
        #         albu.RandomContrast(p=1),
        #         albu.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
        albu.Normalize(),
        ToTensorV2(),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(512, 512),
        albu.RandomCrop(height=512, width=512, always_apply=True),
        albu.Normalize(),
        ToTensorV2(),

    ]
    return albu.Compose(test_transform)

# def to_tensor(x, **kwargs):
#     return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=ToTensorV2, mask=ToTensorV2),
    ]
    return albu.Compose(_transform)