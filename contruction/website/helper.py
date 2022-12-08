import base64
import json
from datetime import datetime
import datetime
from io import BytesIO

import cv2
import pandas as pd
from PIL import Image
from dateutil.relativedelta import relativedelta


class PdEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

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


def get_all_log(now, collection, no_id_and_image=True, start=3, end=-1):
    start_date = now - relativedelta(days=start)
    if end == -1:
        end_date = now
    else:
        end_date = now - relativedelta(days=end)

    assert end_date - start_date >= datetime.timedelta(days = 0)

    res = collection.aggregate([
        {
            '$match': {
                'timestamp': {"$lt": end_date,
                              "$gte": start_date}
                # 'timestamp': same_time
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


def get_a_record(name, timestamp, collection):
    result = collection.aggregate([
        {
            '$match': {
                'name': name,
                'timestamp': timestamp
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


def get_latest_timestamp(collection):
    result = collection.find


