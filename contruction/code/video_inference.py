import time

import cv2
import numpy as np
import pafy
import torch
from PIL import Image

from dataset import canny_preprocess
from utils import load_checkpoint
from model import UNET

from config import DEVICE, IMAGE_HEIGHT, IMAGE_WIDTH, CHECKPOINT_INPUT_PATH, CHECKPOINT_OUTPUT_PATH
import albumentations as A
from albumentations.pytorch import ToTensorV2

model = UNET(in_channels=3, out_channels=1).to(DEVICE)

check_point_path = r"E:\DATN_local\1_IN_USED_CHECKPOINTS\UNET_WITH_RESIDUAL_CLASSIFICATION_PREPROCESSING.pth.tar"

load_checkpoint(torch.load(check_point_path), model)

transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)


label = {
    0: "No water",
    1: "There's shallow water, not affected your route.",
    2: "There's deep water, not recommend to enter the area!",
    3: "Dangerous water in the way!"
}

online = False

# capture = cv2.VideoCapture(0)
capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\2_timelapse_28092022.mp4")
capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\0_extracted_DANANG_STREET_FLOOD_Media1.mp4")

if online:
    url = "https://www.youtube.com/watch?v=1M2IE21aUy4"
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


font = cv2.FONT_HERSHEY_SIMPLEX

prev_time = 0
alpha = 0.9
with torch.no_grad():
    while True:
        check, frame = capture.read()

        if check == True:

            # Display the resulting frame
            # cv2.imshow('Frame', frame)

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = canny_preprocess(image)
            image = transform(image=image)['image']
            image = image.to(device=DEVICE)
            image = image.unsqueeze(0)
            #
            now_time = time.time()
            prediction_mask, prediction_class = model(image)
            inference_time = time.time() - now_time
            fps = int(1/inference_time)

            prediction_mask = torch.sigmoid(prediction_mask)
            prediction_mask = (prediction_mask > 0.5).float()
            prediction_mask = np.array(prediction_mask.cpu())

            prediction_mask = prediction_mask.squeeze(1).reshape((512, 512, 1))

            gray_mask = cv2.cvtColor(prediction_mask*255,cv2.COLOR_GRAY2RGB).astype(np.uint8)
            frame = frame.astype(np.uint8)
            gray_mask = cv2.resize(gray_mask, frame.shape[1::-1])

            dst = cv2.addWeighted(frame, alpha , gray_mask, 1-alpha, 0)

            class_pred = int(torch.argmax(prediction_class, dim=1).cpu())

            height, witdh, channel = dst.shape
            textsize = cv2.getTextSize(label[class_pred], font, 0.5, 1)[0]
            textX = (dst.shape[1] - textsize[0]) // 2
            textY = (dst.shape[0] + textsize[1]) // 2
            cv2.putText(dst, "Advice: "+ label[class_pred], (textX, textY), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(dst, "FPS: "+str(fps), (7, 70), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Flood Detection", dst)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break



    # Break the loop
    # else:
    #     print("stop running at: ", time.time(), cnt)
    #     break

# When everything done, release the video capture object
capture.release()

