import argparse
import csv
import os
from datetime import datetime

import cv2
import numpy as np
import pafy
import segmentation_models_pytorch as smp
import torch

from model_zoo.augmentation import get_validation_augmentation
from model_zoo.model import UNET
from model_zoo.utils import load_checkpoint

# python inference_app.py -e mobilenet_v2 -p "E:\DATN_local\1_IN_USED_CHECKPOINTS\drive-download-20221115T041425Z-001\mobilenet_v2_imagenet_1.pth.tar" -o True -u "https://www.youtube.com/watch?v=-y6ql2bacSo" -csv "E:\DATN_local\2_HISTORY_INFERENCE" -n TEST

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-e', "--encoder", type=str, default='mobilenet_v2')
parser.add_argument('-w', "--weight", type=str, default='imagenet')
parser.add_argument('-p', '--path', type=str, default='')
parser.add_argument("-v", '--version', type=str, default='1')
parser.add_argument('-d', '--device', type=str, default='cuda')
parser.add_argument('-o', '--online', type=bool, default=False)
parser.add_argument('-u', '--url', type=str, default='')
parser.add_argument('-csv', '--csv', type=str)
parser.add_argument('-n', '--name', type=str)

args = parser.parse_args()

def write_log(path_to_csv, value, debug=False):

    with open(path_to_csv, 'a', newline='',) as f:
        writer = csv.writer(f)
        writer.writerow(value)
        if debug:
            print(f"append log {path_to_csv}:",value)

if __name__ == "__main__":

    ENCODER = args.encoder
    DEVICE = args.device
    online = args.online
    check_point_path = args.path
    video_collect = False
    url = args.url
    live_name = args.name
    csv_path = args.csv + os.sep + ENCODER + "_" + live_name +".json"
    debug = False
    total_pixel = 512*512

    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        activation=None,  # activation function, default is None
        classes=4,  # define number of output labels
    )
    if "my_unet" in ENCODER:
        model = UNET(in_channels=3, out_channels=1)
    else:
        model = smp.Unet(encoder_name=ENCODER, encoder_weights='imagenet', classes=1, aux_params=aux_params)
    model.to(DEVICE)

    load_checkpoint(torch.load(check_point_path), model)

    transform = get_validation_augmentation()

    label = {
        0: "No water",
        1: "There's shallow water, not affected your route.",
        2: "There's deep water, not recommend to enter the area!",
        3: "Dangerous water in the way!"
    }

    if online:
        # url = "https://www.youtube.com/watch?v=1M2IE21aUy4"
        # url = "https://www.youtube.com/watch?v=fiWopDJ3rCs"
        video = pafy.new(url)
        best = video.getbest()
        streams = video.streams

        for s in streams:
            # print(s.resolution, s.extension, s.get_filesize(), s.url)
            if s.resolution == '1920x1080':
                url_fullhd = s.url
                break

        if best.url is not None:
            url_fullhd = best.url
        capture = cv2.VideoCapture(url_fullhd)

    else:
        capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\0_extracted_DANANG_STREET_FLOOD_Media1.mp4")

    prev_time = 0
    alpha = 0.5

    if video_collect:
        width = capture.get(3)  # float `width`
        height = capture.get(4) # float `height`
        cnt = 0
        max_length = 100
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f'{ENCODER}_norain.avi', fourcc, 25, (int(width), int(height)))

    with torch.no_grad():
        while True:
            check, frame = capture.read()

            if check == True:

                start_postprocess = torch.cuda.Event(enable_timing=True)
                end_postprocess = torch.cuda.Event(enable_timing=True)

                start_preprocess = torch.cuda.Event(enable_timing=True)
                end_preprocess = torch.cuda.Event(enable_timing=True)

                start_inference = torch.cuda.Event(enable_timing=True)
                end_inference = torch.cuda.Event(enable_timing=True)

                # Display the resulting frame
                # cv2.imshow('Frame', frame)
                # start_preprocess = time.time()
                start_preprocess.record()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = transform(image=image)['image']
                image = image.to(device=DEVICE)
                image = image.unsqueeze(0)
                end_preprocess.record()
                # fps_preprocess = int(1./(time.time()-(start_preprocess) +1e-10))
                #
                # start_inference = time.time()
                start_inference.record()
                prediction_mask, prediction_class = model(image)
                # inference_time = time.time() - start_inference
                # fps_inference = int(1. / (inference_time+1e-10))
                end_inference.record()
                start_postprocess.record()

                prediction_mask = torch.sigmoid(prediction_mask)
                prediction_mask = (prediction_mask > 0.5).float()
                count_flood_pixel = torch.sum(prediction_mask)
                #------

                prediction_mask = prediction_mask*255
                prediction_mask = prediction_mask.squeeze(1).squeeze(1).repeat(3,1,1).permute(1, 2, 0)

                prediction_mask = np.array(prediction_mask.cpu()).astype(np.uint8)

                # gray_mask = cv2.cvtColor(prediction_mask,cv2.COLOR_GRAY2RGB).astype(np.uint8)
                # dst = prediction_mask

                # frame = frame.astype(np.uint8)
                gray_mask = cv2.resize(prediction_mask, frame.shape[1::-1])
                dst = cv2.addWeighted(frame, alpha , gray_mask, 1-alpha, 0)

                class_pred = int(torch.argmax(prediction_class, dim=1).cpu())
                # fps_postprocess = int(1./(time.time()-start_postprocess+1e-8))
                end_postprocess.record()

                torch.cuda.synchronize()
                fps_preprocess = 1./(start_preprocess.elapsed_time(end_preprocess)/1000)
                fps_inference = 1./(start_inference.elapsed_time(end_inference)/1000)
                fps_postprocess = 1./(start_postprocess.elapsed_time(end_postprocess)/1000)

                time_stamp = datetime.now()
                sofi = (count_flood_pixel / total_pixel).item()
                recording = [time_stamp, class_pred, sofi]

                write_log(csv_path, recording, debug=True)



        # Break the loop
        # else:
        #     print("stop running at: ", time.time(), cnt)
        #     break

    # When everything done, release the video capture object
    capture.release()


