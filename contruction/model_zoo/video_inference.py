import argparse
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
from utils import load_checkpoint

import albumentations as A
from albumentations.pytorch import ToTensorV2

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-e', "--encoder", type=str, default='mobilenet_v2')
parser.add_argument('-w', "--weight", type=str, default='imagenet')
parser.add_argument('-p', '--path', type=str, default='')
parser.add_argument("-v", '--version', type=str, default='1')
parser.add_argument('-d', '--device', type=str, default='cuda')
parser.add_argument('-o', '--online', type=bool, default=False)

args = parser.parse_args()

if __name__ == "__main__":

    ENCODER = args.encoder
    DEVICE = args.device
    online = args.online
    check_point_path = args.path


    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        activation=None,  # activation function, default is None
        classes=4,  # define number of output labels
    )
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
        url = "https://www.youtube.com/watch?v=oAMqoAaxPl4"
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
        # capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\0_extracted_DANANG_STREET_FLOOD_Media1.mp4")
        # capture = cv2.VideoCapture(0)
        # capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\2_timelapse_28092022.mp4")
        capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\4_heavy_flood.mp4")

    total_pixel = 512*512
    font = cv2.FONT_HERSHEY_SIMPLEX

    prev_time = 0
    alpha = 0.9
    fig = plt.figure()
    x1 = np.linspace(0.0, 100.0)
    y1 = [0 for i in range(len(x1))]
    line1, = plt.plot(x1, y1, 'b.-.')
    fig.canvas.draw()
    plot = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
    plot = np.array(plot).astype(np.uint8)
    size = 300
    plot = cv2.resize(plot, (size, size))
    plt2gray = cv2.cvtColor(plot, cv2.COLOR_RGB2GRAY)
    ret, mask_plot = cv2.threshold(plt2gray, 1, 255, cv2.THRESH_BINARY)
    ax = plt.gca()


    with torch.no_grad():
        while True:
            check, frame = capture.read()

            if check == True:

                # Display the resulting frame
                # cv2.imshow('Frame', frame)
                start_preprocess = time.time()
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = transform(image=image)['image']
                image = image.to(device=DEVICE)
                image = image.unsqueeze(0)
                fps_preprocess = int(1/(time.time()-start_preprocess))
                #
                start_inference = time.time()
                prediction_mask, prediction_class = model(image)
                inference_time = time.time() - start_inference
                fps_inference = int(1 / inference_time)

                start_postprocess = time.time()
                prediction_mask = torch.sigmoid(prediction_mask)
                prediction_mask = (prediction_mask > 0.5).float()
                count_flood_pixel = prediction_mask.sum()
                prediction_mask = prediction_mask.squeeze(1).reshape((512, 512, 1))
                prediction_mask = np.array(prediction_mask.cpu())
                gray_mask = cv2.cvtColor(prediction_mask*255,cv2.COLOR_GRAY2RGB).astype(np.uint8)
                frame = frame.astype(np.uint8)
                gray_mask = cv2.resize(gray_mask, frame.shape[1::-1])
                dst = cv2.addWeighted(frame, alpha , gray_mask, 1-alpha, 0)

                class_pred = int(torch.argmax(prediction_class, dim=1).cpu())
                fps_postprocess = int(1/(time.time()-start_postprocess))

                height, witdh, channel = dst.shape
                textsize = cv2.getTextSize(label[class_pred], font, 0.5, 1)[0]
                textX = (dst.shape[1] - textsize[0]) // 2
                textY = (dst.shape[0] + textsize[1]) // 2
                cv2.putText(dst, "Advice: "+ label[class_pred], (textX, textY), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dst, "Pre-processing FPS: "+str(fps_preprocess), (7, 60), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dst, "Inference FPS: " + str(fps_inference), (7, 80), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(dst, "Post-processing FPS: "+str(fps_postprocess), (7, 100), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                sofi = (count_flood_pixel/total_pixel).item()
                y1.pop(0)
                y1.append(sofi)
                line1.set_ydata(y1)
                ax.set_ylim([0, 1])
                ax.autoscale_view()
                fig.canvas.draw()
                plot = PIL.Image.frombytes('RGB', fig.canvas.get_width_height(),fig.canvas.tostring_rgb())
                plot = np.array(plot).astype(np.uint8)
                # img is rgb, convert to opencv's default bgr
                plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
                plot = cv2.resize(plot, (size, size))
                roi = dst[-size-10:-10, -size-10:-10]
                roi[np.where(mask_plot)]=0
                roi += plot


                cv2.imshow("Flood Detection", dst)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break



        # Break the loop
        # else:
        #     print("stop running at: ", time.time(), cnt)
        #     break

    # When everything done, release the video capture object
    capture.release()

