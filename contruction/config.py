import datetime
import os

import torch

from model_zoo.augmentation import get_validation_augmentation
from model_zoo.model import UNET
import segmentation_models_pytorch as smp

from model_zoo.utils import load_checkpoint

def create_dir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

stream_links = ["https://www.youtube.com/watch?v=-y6ql2bacSo", "https://www.youtube.com/watch?v=fiWopDJ3rCs", "https://www.youtube.com/watch?v=3zH2GwsexiE"]
names = ["BENHVIENC", "NGUYENHUESCHOOL", "PHUONGTRAN"]
ENCODER = "mobilenet_v2"
DEVICE = 'cuda'
check_point_path = r"E:\DATN_local\1_IN_USED_CHECKPOINTS\mobilenet_v2_imagenet_5.pth.tar"
trace_folder = r"E:\DATN_local\2_TORCH_TRACE_MODELS"
history_folder = r"E:\DATN_local\2_HISTORY_INFERENCE"
logging_frequency = 2 #second



csv_log_file = history_folder + os.sep + ENCODER + "_" + str(datetime.date.today()) + ".json"

create_dir(history_folder + os.sep + str(datetime.date.today()))

history_image_log_folder = []
for i in range(len(names)):
    history_image_log_folder.append(history_folder + os.sep + str(datetime.date.today()) + os.sep + names[i])
    create_dir(history_image_log_folder[i])



aux_params = dict(
    pooling='avg',  # one of 'avg', 'max'
    activation=None,  # activation function, default is None
    classes=4,  # define number of output labels
)

if "my_unet" in ENCODER:
    model = UNET(in_channels=3, out_channels=1)
else:
    model = smp.Unet(encoder_name=ENCODER, classes=1, aux_params=aux_params)

load_checkpoint(torch.load(check_point_path), model)
model.to(DEVICE)
transform = get_validation_augmentation()