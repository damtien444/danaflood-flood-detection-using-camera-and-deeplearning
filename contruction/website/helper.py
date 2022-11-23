import base64
from datetime import datetime
from io import BytesIO

import cv2
import pandas as pd
from PIL import Image


def image_encoding(cv_image):
    retval, buffer = cv2.imencode('.jpg', cv_image)
    jpg_as_text = base64.b64encode(buffer)
    return jpg_as_text


def get_list_cam_log(name, collection, no_id_and_image=True):
    result = collection.aggregate([
        {
            '$match': {
                'name': name
            }
        }, {
            '$project': {
                'timestamp': 1,
                'warning_index': 1,
                'name': 1,
                'sofi': 1
            }
        }
    ])

    df = pd.DataFrame(list(result))
    if len(df) > 0:
        if no_id_and_image:
            del df['_id']
    return df


def get_all_log(collection, no_id_and_image=True):
    res = collection.aggregate([
        {
            '$project': {
                'timestamp': 1,
                'warning_index': 1,
                'name': 1,
                'sofi': 1
            }
        }
    ])

    df = pd.DataFrame(list(res))
    if len(df) > 0:
        if no_id_and_image:
            del df['_id']
    return df


def image_decoding(b64):
    im_bytes = base64.b64decode(b64)  # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)
    return img


def get_last_record(name, collection):
    result = collection.aggregate([
        {
            '$match': {
                'name': name
            }
        }, {
            '$sort': {
                'timestamp': -1
            }
        }, {
            '$limit': 1
        }
    ])

    res = list(result)
    if len(res) > 0:
        record = res[0]
        record['image_b64'] = image_decoding(record['image_b64'])
        return record
    return res


def get_latest_unique_warning(collection, no_id_and_image=True):
    result = collection.aggregate([
        {
            '$match': {
                'warning_index': {
                    '$gt': 1
                }
            }
        },
        {
            '$project': {
                'timestamp': 1,
                'warning_index': 1,
                'name': 1,
                'sofi': 1
            }
        },
        {
            '$sort': {
                'timestamp': -1
            }
        }, {
            '$group': {
                '_id': '$name',
                'doc': {
                    '$first': '$$ROOT'
                }
            }
        }, {
            '$replaceRoot': {
                'newRoot': '$doc'
            }
        }
    ])


    df = pd.DataFrame(list(result))
    if len(df)>0:
        if no_id_and_image:
            del df['_id']
    return df

#
def insert_log(collection, name, cls_index, sofi, image):
    record = {
        'name': name,
        'timestamp': datetime.today().replace(microsecond=0),
        'warning_index': cls_index,
        'sofi': sofi,
        'image_b64': image
    }

    return collection.insert_one(record)
