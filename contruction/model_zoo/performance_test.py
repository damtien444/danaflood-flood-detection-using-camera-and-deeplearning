import argparse
import json
import os
import time

import PIL
import cv2
import numpy as np
import pafy
import torch
from PIL import Image
from matplotlib import pyplot as plt
import segmentation_models_pytorch as smp
from augmentation import get_validation_augmentation
from model import UNET
from utils import load_checkpoint
import torchvision.transforms as T

import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-p', '--path', type=str, default='')
parser.add_argument('-d', '--device', type=str, default='cuda')

args = parser.parse_args()

def numel(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)

if __name__ == "__main__":

    # ENCODER = args.encoder
    # DEVICE = args.device
    # online = args.online
    check_point_path = args.path
    DEVICE = args.device

    with open('fps_results.json', 'r') as f:
        test_result = json.load(f)
        print("reload")

    for file in os.listdir(check_point_path):

        index_keyword = file.index("_imagenet")

        ENCODER = file.split(".")[0][:index_keyword]

        path = check_point_path + "\\" + file

        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            activation=None,  # activation function, default is None
            classes=4,  # define number of output labels
        )
        if "my_unet" in file:
            model = UNET(in_channels=3, out_channels=1)
        else:
            model = smp.Unet(encoder_name=ENCODER, encoder_weights='imagenet', classes=1, aux_params=aux_params)

        model.to(DEVICE)

        # load_checkpoint(torch.load(path), model)

        total_params = numel(model)
        test_result["number_param_"+ENCODER] = total_params
        print(ENCODER, total_params)

        transform = get_validation_augmentation()

        capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\0_extracted_DANANG_STREET_FLOOD_Media1.mp4")
        capture = cv2.VideoCapture(0)
        capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\2_timelapse_28092022.mp4")
        capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\4_heavy_flood.mp4")

        cnt = 0
        max_cnt = 100
        start_test = time.time()
        time_inference = []

        with torch.no_grad():
            while True:
                check, frame = capture.read()

                if check == True:

                    cnt+=1

                    # Display the resulting frame
                    # cv2.imshow('Frame', frame)
                    # start_preprocess = time.time()
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = transform(image=image)['image']
                    image = image.to(device=DEVICE)
                    image = image.unsqueeze(0)
                    # fps_preprocess = int(1/(time.time()-start_preprocess + 1e-8))
                    #
                    start_inference = time.time()
                    prediction_mask, prediction_class = model(image)
                    inference_time = time.time() - start_inference
                    fps_inference = time_inference.append(inference_time)

                    if cnt > max_cnt:
                        break

        test_result[ENCODER+"_"+DEVICE+"_fps"] =  [sum(time_inference) / len(time_inference), len(time_inference)/sum(time_inference)]
        print(f"{ENCODER}\t{sum(time_inference) / len(time_inference)}\t{len(time_inference)/sum(time_inference) }")

        # When everything done, release the video capture object
        capture.release()

    with open('param_number_results.json', 'w') as f:
        json.dump(test_result, f)
        print("save_test_result")


