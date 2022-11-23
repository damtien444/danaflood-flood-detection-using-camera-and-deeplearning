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

# stream_links = ["https://www.youtube.com/watch?v=-y6ql2bacSo", "https://www.youtube.com/watch?v=fiWopDJ3rCs", "https://www.youtube.com/watch?v=3zH2GwsexiE"]
# names = ["BENHVIENC", "NGUYENHUESCHOOL", "PHUONGTRAN"]

# is_offline, link/path, name
input_camera_list = [
    (False, "https://www.youtube.com/watch?v=-y6ql2bacSo", "BENHVIENC"),
    (False, "https://www.youtube.com/watch?v=fiWopDJ3rCs", "NGUYENHUESCHOOL"),
    (False, "https://www.youtube.com/watch?v=3zH2GwsexiE", "PHUONGTRAN"),
    # (True, r"E:\DATN_local\self_collected_data\4_heavy_flood.mp4", "Fast flood site"),
    (True, r"E:\DATN_local\3_DEMO\Site_1 - Made with Clipchamp (1).mp4", "CAM_SITE_1"),
    # (True, r"E:\DATN_local\3_DEMO\Site_2 - Made with Clipchamp.mp4", "CAM_SITE_2"),
    # (True, r"E:\DATN_local\3_DEMO\Site_3 - Made with Clipchamp.mp4", "CAM_SITE_3"),
]

# names = ["STREET_FLOOD", "LIVE"]
files = [r"E:\DATN_local\self_collected_data\0_extracted_DANANG_STREET_FLOOD_Media1.mp4", r"E:\DATN_local\self_collected_data\1_live_record_video_092022.mp4"]
ENCODER = "mobilenet_v2"
DEVICE = 'cuda'
check_point_path = r"E:\DATN_local\1_IN_USED_CHECKPOINTS\mobilenet_v2_imagenet_6.pth.tar"
trace_folder = r"E:\DATN_local\2_TORCH_TRACE_MODELS"
history_folder = r"E:\DATN_local\2_HISTORY_INFERENCE"
logging_frequency = 10 #second
batch_process_size = 2


csv_log_file = history_folder + os.sep + ENCODER + "_" + str(datetime.date.today()) + ".json"

create_dir(history_folder + os.sep + str(datetime.date.today()))

history_image_log_folder = []
for i in range(len(input_camera_list)):
    history_image_log_folder.append(history_folder + os.sep + str(datetime.date.today()) + os.sep + input_camera_list[i][2])
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

# phải có model eval để batch chạy đúng
model.eval()
load_checkpoint(torch.load(check_point_path), model)
model.to(DEVICE)
transform = get_validation_augmentation()