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
from model import UNET
from utils import load_checkpoint
import torchvision.transforms as T

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
    video_collect = True


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

    model.eval()

    load_checkpoint(torch.load(check_point_path), model)

    transform = get_validation_augmentation()

    label = {
        0: "Level 0",
        1: "Level 1",
        2: "Level 2",
        3: "Level 3"
    }



    if online:
        # url = "https://www.youtube.com/watch?v=1M2IE21aUy4"
        # url = "https://www.youtube.com/watch?v=fiWopDJ3rCs"
        url = "https://www.youtube.com/watch?v=-y6ql2bacSo"
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
        # capture = cv2.VideoCapture(r"E:\DATN_local\3_DEMO\Site_1 - Made with Clipchamp (1).mp4")
        # capture = cv2.VideoCapture(r"E:\DATN_local\3_DEMO\Site_2 - Made with Clipchamp.mp4")
        capture = cv2.VideoCapture(r"E:\DATN_local\3_DEMO\Site_3 - Made with Clipchamp.mp4")
        # capture = cv2.VideoCapture(r"E:\DATN_local\3_DEMO\Site_1 - Made with Clipchamp (1).mp4")
        # capture = cv2.VideoCapture(0)
        # capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\2_timelapse_28092022.mp4")
        # capture = cv2.VideoCapture(r"E:\DATN_local\self_collected_data\4_heavy_flood.mp4")

    total_pixel = 512*512
    font = cv2.FONT_HERSHEY_SIMPLEX

    prev_time = 0
    alpha = 0.75
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

    if video_collect:
        width = capture.get(3)  # float `width`
        height = capture.get(4) # float `height`
        cnt = 0
        max_length = 1000
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(f'demo_{ENCODER}_streetflood_3.avi', fourcc, 50, (int(width), int(height)))

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

                height, witdh, channel = dst.shape
                textsize = cv2.getTextSize(label[class_pred], font, 3.5, 10)[0]
                textX = (dst.shape[1] - textsize[0]) // 2
                textY = (dst.shape[0] + textsize[1]) // 2
                cv2.putText(dst, label[class_pred], (textX, textY), font, 3.5, (255, 255, 255), 10, cv2.LINE_AA)
                cv2.putText(dst, "Pre-processing FPS: "+str(fps_preprocess), (7, height-150), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(dst, "Inference FPS: " + str(fps_inference), (7, height-100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(dst, "Post-processing FPS: "+str(fps_postprocess), (7, height-50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

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

                if video_collect:
                    video.write(dst)
                    cnt += 1
                    if cnt > max_length:
                        break

                cv2.imshow("Flood Detection", dst)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break



        # Break the loop
        # else:
        #     print("stop running at: ", time.time(), cnt)
        #     break

    # When everything done, release the video capture object
    capture.release()
    if video_collect:
        video.release()

