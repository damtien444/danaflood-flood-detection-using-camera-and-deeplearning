import csv
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import pafy
import torch

from config import model, stream_links, transform, DEVICE, csv_log_file, names, history_image_log_folder, \
    logging_frequency

alpha = 0.5

def get_captures(stream_links):
    # url = "https://www.youtube.com/watch?v=1M2IE21aUy4"
    # url = "https://www.youtube.com/watch?v=fiWopDJ3rCs"
    captures = []
    for url in stream_links:
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
        capture = VideoCapture(url_fullhd)
        captures.append(capture)

    return captures

def transformation(image):

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = transform(image=image)['image']
    image = image.to(device=DEVICE)
    image = image.unsqueeze(0)
    return image

def write_log(path_to_csv, value, debug=False):

    with open(path_to_csv, 'a', newline='',) as f:
        writer = csv.writer(f)
        writer.writerow(value)
        if debug:
            print(f"append log {path_to_csv}:",value)

class VideoCapture:
  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  # read frames as soon as they are available, keeping only most recent one
  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()   # discard previous (unprocessed) frame
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()


class frequecy_average:
    def __init__(self):
        self.cnt = 0
        self.sofis = []
        self.clses = []
        self.start = time.time()

    def update(self, sofi, clses):
        self.cnt+=1
        self.sofis.append(sofi)
        self.clses.append(clses)

    def reset(self):
        self.cnt = 0
        self.sofis = []
        self.clses = []
        self.start = time.time()

    def get(self):
        ret = sum(self.sofis)/(self.cnt+1e-8), sum(self.clses)/(self.cnt+1e-8)
        self.reset()
        return ret


def runner(model, stream_links):
    captures = get_captures(stream_links)
    logs = []
    for i in range(len(captures)):
        logs.append(frequecy_average())


    with torch.no_grad():
        while True:
            origin_frame = []
            images = []
            ids = []
            for idx, capture in enumerate(captures):
                frame = capture.read()
                image = transformation(frame)
                frame = cv2.resize(frame, (512,512), interpolation=cv2.INTER_AREA)
                images.append(image)
                origin_frame.append(frame)
                ids.append(idx)

            if len(images)>0:
                images = torch.cat(images)
                images.to(DEVICE)

                prediction_masks, prediction_class = model(images)


                prediction_masks = torch.sigmoid(prediction_masks)
                prediction_masks = (prediction_masks > 0.5).float()

                prediction_masks = prediction_masks.reshape((len(images),512,512))
                class_preds = (torch.argmax(prediction_class, dim=1))

                for i in range(len(images)):
                    single_image_mask = prediction_masks[i]
                    single_image_cls = class_preds[i]

                    sofi = torch.sum(single_image_mask) / (512*512.)
                    # ------

                    single_image_mask = single_image_mask * 255
                    single_image_mask = single_image_mask.squeeze(1).squeeze(1).repeat(3, 1, 1).permute(1, 2, 0)

                    single_image_mask = np.array(single_image_mask.cpu()).astype(np.uint8)

                    dst = cv2.addWeighted(np.array(origin_frame[i]).astype(np.uint8), alpha, single_image_mask, 1 - alpha, 0)
                    # dst = cv2.addWeighted(np.array(images[i].permute(1, 2, 0).cpu()).astype(np.uint8), alpha, single_image_mask, 1 - alpha, 0)

                    logs[ids[i]].update(sofi.item(), single_image_cls.item())

                    if time.time() - logs[ids[i]].start > logging_frequency:
                        sofi, cls = logs[ids[i]].get()

                        write_log(csv_log_file, [names[ids[i]], datetime.now(), cls, sofi])

                        file_name = names[ids[i]]+"_"+datetime.now().strftime("%H_%M_%S")+".jpg"

                        cv2.imwrite(history_image_log_folder[i]+os.sep + file_name, dst)



if __name__=='__main__':
    runner(model, stream_links)