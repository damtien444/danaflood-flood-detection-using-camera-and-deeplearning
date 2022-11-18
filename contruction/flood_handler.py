import io

import torch
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
import torch.nn.functional as F

from model_zoo.augmentation import get_validation_augmentation


#
class FloodHandler(BaseHandler):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = get_validation_augmentation()

    def preprocess_one_image(self, req):
        """
        Process one single image.
        """
        # get image from the request
        image = req.get("data")
        if image is None:
            image = req.get("body")
         # create a stream from the encoded image
        image = Image.open(io.BytesIO(image))
        image = self.transform(image)
        # add batch dim
        image = image.unsqueeze(0)
        return image

    def preprocess(self, requests):
        """
        Process all the images from the requests and batch them in a Tensor.
        """
        images = [self.preprocess_one_image(req) for req in requests]
        images = torch.cat(images)
        return images

    def inference(self, data, *args, **kwargs):
        """
        Given the data from .preprocess, perform inference using the model.
        We return the predicted label for each image.
        """
        prediction_mask, cls = self.model.forward(data)
        prediction_mask = torch.sigmoid(prediction_mask)
        prediction_mask = (prediction_mask > 0.5).float()

        probs = F.softmax(cls, dim=1)
        preds = torch.argmax(probs, dim=1)
        return prediction_mask, preds

    def postprocess(self, data):
        """
                Given the data from .inference, postprocess the output.
                In our case, we get the human readable label from the mapping
                file and return a json. Keep in mind that the reply must always
                be an array since we are returning a batch of responses.
                """
        masks, labels = data
        res = []
        # pres has size [BATCH_SIZE, 1]
        # convert it to list
        preds = labels.cpu().tolist()
        masks = masks.cpu().tolist()

        for idx, pred in enumerate(preds):
            mask = masks[idx]
            count_flood_pixel = torch.sum(mask)
            sofi = count_flood_pixel / (512*512.)
            label = self.mapping[str(pred)][1]
            res.append({'label': label, 'index': pred, 'mask': mask, 'sofi': sofi})
        return res

_service = FloodHandler()

def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    data = _service.preprocess(data)
    data = _service.inference(data)
    data = _service.postprocess(data)

    return data