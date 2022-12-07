import base64
import csv
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pafy
import pymongo
import torch

from config import model, transform, DEVICE, \
    logging_frequency, files, input_camera_list, batch_process_size

client = pymongo.MongoClient("mongodb+srv://FLOODING_PROTOTYPE:FLOODING@cluster0.v1qjbym.mongodb.net/?retryWrites=true&w=majority")
db = client['danang_flood']
collection = db['camera_logging']

alpha = 0.5


def get_captures(input, update_index=None, captures=None):
    # url = "https://www.youtube.com/watch?v=1M2IE21aUy4"
    # url = "https://www.youtube.com/watch?v=fiWopDJ3rCs"
    if update_index == None:
        captures = []
        for idx, (type, stream_link, name) in enumerate(input):
            if not type:
                try:
                    video = pafy.new(stream_link)
                except:
                    captures.append((idx, None))
                    continue

                best = video.getbest()
                streams = video.streams

                for s in streams:
                    # print(s.resolution, s.extension, s.get_filesize(), s.url)
                    if s.resolution == '1280x720':
                        url_fullhd = s.url
                        break


                capture = cv2.VideoCapture(url_fullhd)
                captures.append((idx, capture))
            else:
                capture = cv2.VideoCapture(stream_link)
                captures.append((idx, capture))

        return captures
    else:
        for idx in update_index:
            (type, stream_link, name) = input[idx]
            try:
                video = pafy.new(stream_link)
            except:
                captures[idx] = (idx, None)
                continue
            best = video.getbest()
            streams = video.streams

            for s in streams:
                # print(s.resolution, s.extension, s.get_filesize(), s.url)
                if s.resolution == '1280x720':
                    url_fullhd = s.url
                    break

            capture = cv2.VideoCapture(url_fullhd)
            captures[idx] = (idx, capture)
        return captures


def transformation(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    image = image.to(device=DEVICE)
    image = image.unsqueeze(0)
    return image


def write_log(path_to_csv, value, debug=False):
    with open(path_to_csv, 'a', newline='', ) as f:
        writer = csv.writer(f)
        writer.writerow(value)
        if debug:
            print(f"append log {path_to_csv}:", value)




class frequecy_average:
    def __init__(self):
        self.cnt = 0
        self.sofis = []
        self.clses = []
        self.start = time.time()

    def update(self, sofi, clses):
        self.cnt += 1
        self.sofis.append(sofi)
        self.clses.append(clses)

    def reset(self):
        self.cnt = 0
        self.sofis = []
        self.clses = []
        self.start = time.time()

    def get(self):
        ret = sum(self.sofis) / (self.cnt + 1e-8), sum(self.clses) / (self.cnt + 1e-8)
        self.reset()
        return ret

def insert_log(name, cls_index, sofi, image):
    record ={
        'name': name,
        'timestamp': datetime.today().replace(microsecond=0),
        'warning_index':cls_index,
        'sofi':sofi,
        'image_b64': image
    }

    return collection.insert_one(record)

def image_encoding(cv_image):
    retval, buffer = cv2.imencode('.jpg', cv_image)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text

def producer_runner(queue):
    # get_capture
    # captures = get_captures(files, is_offline=True)
    captures = get_captures(input_camera_list)
    print(len(captures))

    # latest_frame = []
    # for (idx, capture) in captures:
        # latest_object = TakeCameraLatestPictureThread(capture)
        # latest_frame.append((idx, latest_object))

    start = []

    for i in range(len(captures)):
        start.append(0)

    try_ = []

    for i in range(len(captures)):
        try_.append(True)

    while True:
        # time.sleep(logging_frequency/10)

        batch_frame = []
        batch_image = []
        batch_idx = []
        to_be_update_capture = []
        for (idx, capture) in captures:

            # frame = capture.frame
            # chưa cover trường hợp livestream bị not available -> bị dừng live thì các camera khác phải hoạt động bình thường
            # todo: Viết một hàm evaluate stream health/status
            if capture is None:
                continue

            _, frame = capture.read()
            #
            time.sleep(0.05) # read 24 frame
            now = time.time()
            #
            if now - start[idx] < (logging_frequency // 10):
                continue

            if try_[idx] == False and _ == False:
                # retry to get renew capture object
                to_be_update_capture.append(idx)

            try_[idx] = _

            print(idx, _)
            start[idx] = time.time()

            if frame is not None:
                image = transformation(frame)
                frame = cv2.resize(frame, (512,512), interpolation=cv2.INTER_AREA)
                batch_image.append(image)
                batch_frame.append(frame)
                batch_idx.append(idx)

        if len(to_be_update_capture) > 0:
            captures = get_captures(input_camera_list, to_be_update_capture, captures)

        # print(len(batch_frame), len(batch_image), batch_idx)

        if len(batch_idx)>0:
            print("producer send:", len(batch_image))
            queue.put((batch_frame, batch_image, batch_idx))
        # iterate through capture, get latest images

        # batch images into queue -> batch_size

def consumer_runner(model, queue):
    # captures = get_captures(stream_links)

    names = []
    for idx, (type, stream_link, name) in enumerate(input_camera_list):
        names.append(name)

    logs = []
    for i in range(len(names)):
        logs.append(frequecy_average())

    with torch.no_grad():
        while True:

            origin_frames, images, ids = queue.get()
            print("consumer receive:", len(origin_frames))

            if len(images) > batch_process_size:
                for i in range(0, len(images), batch_process_size):
                    print('consumer inference:', i, i+batch_process_size)
                    batch_image = images[i:i+batch_process_size]
                    batch_ids = ids[i:i+batch_process_size]
                    batch_frames = origin_frames[i:i+batch_process_size]
                    batch_image = torch.cat(batch_image)
                    batch_image.to(DEVICE)

                    prediction_masks, prediction_class = model(batch_image)

                    prediction_masks = torch.sigmoid(prediction_masks)
                    prediction_masks = (prediction_masks > 0.5).float()

                    prediction_masks = prediction_masks.reshape((len(batch_image), 512, 512))
                    class_preds = (torch.argmax(prediction_class, dim=1))

                    for j in range(len(batch_image)):
                        single_image_mask = prediction_masks[j]
                        single_image_cls = class_preds[j]

                        sofi = torch.sum(single_image_mask) / (512. * 512.)
                        # ------

                        single_image_mask = single_image_mask * 255
                        single_image_mask = single_image_mask.squeeze(1).squeeze(1).repeat(3, 1, 1).permute(1, 2, 0)

                        single_image_mask = np.array(single_image_mask.cpu()).astype(np.uint8)

                        dst = cv2.addWeighted(np.array(batch_frames[j]).astype(np.uint8), alpha, single_image_mask,
                                              1 - alpha, 0)
                        # dst = cv2.addWeighted(np.array(images[i].permute(1, 2, 0).cpu()).astype(np.uint8), alpha, single_image_mask, 1 - alpha, 0)

                        logs[batch_ids[j]].update(sofi.item(), single_image_cls.item())

                        if time.time() - logs[batch_ids[j]].start > logging_frequency:
                            print("LOG", logs[batch_ids[j]].start)
                            sofi, cls = logs[batch_ids[j]].get()
                            insert_log(names[batch_ids[j]], cls, sofi, image_encoding(dst))
            elif len(images) > 0:
                print('consumer inference:', len(images))
                images = torch.cat(images)
                images.to(DEVICE)

                prediction_masks, prediction_class = model(images)

                prediction_masks = torch.sigmoid(prediction_masks)
                prediction_masks = (prediction_masks > 0.5).float()

                prediction_masks = prediction_masks.reshape((len(images), 512, 512))
                class_preds = (torch.argmax(prediction_class, dim=1))

                for i in range(len(images)):
                    single_image_mask = prediction_masks[i]
                    single_image_cls = class_preds[i]

                    sofi = torch.sum(single_image_mask) / (512. * 512.)
                    # ------

                    single_image_mask = single_image_mask * 255
                    single_image_mask = single_image_mask.squeeze(1).squeeze(1).repeat(3, 1, 1).permute(1, 2, 0)

                    single_image_mask = np.array(single_image_mask.cpu()).astype(np.uint8)

                    dst = cv2.addWeighted(np.array(origin_frames[i]).astype(np.uint8), alpha, single_image_mask,
                                          1 - alpha, 0)
                    # dst = cv2.addWeighted(np.array(images[i].permute(1, 2, 0).cpu()).astype(np.uint8), alpha, single_image_mask, 1 - alpha, 0)

                    logs[ids[i]].update(sofi.item(), single_image_cls.item())
                    if time.time() - logs[ids[i]].start > logging_frequency:
                        print("LOG", logs[ids[i]].start)
                        sofi, cls = logs[ids[i]].get()
                        insert_log(names[ids[i]],  cls, sofi, image_encoding(dst))
                        # write_log(csv_log_file, [names[ids[i]], datetime.now(), cls, sofi])
                        #
                        # file_name = names[ids[i]] + "_" + datetime.now().strftime("%H_%M_%S") + ".jpg"
                        #
                        # cv2.imwrite(history_image_log_folder[ids[i]] + os.sep + file_name, dst)



if __name__ == '__main__':
    queue = queue.Queue()

    consumer = threading.Thread(target=consumer_runner, args=(model,queue, ))
    consumer.start()

    producer = threading.Thread(target=producer_runner, args=(queue,))
    producer.start()


