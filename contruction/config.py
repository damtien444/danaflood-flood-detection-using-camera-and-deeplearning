from model_zoo.model import UNET
import segmentation_models_pytorch as smp

stream_links = []


ENCODER = "vgg13"
DEVICE = 'cuda'


aux_params = dict(
    pooling='avg',  # one of 'avg', 'max'
    activation=None,  # activation function, default is None
    classes=4,  # define number of output labels
)

if "my_unet" in ENCODER:
    model = UNET(in_channels=3, out_channels=1)
else:
    model = smp.Unet(encoder_name=ENCODER, classes=1, aux_params=aux_params)

model.to(DEVICE)

