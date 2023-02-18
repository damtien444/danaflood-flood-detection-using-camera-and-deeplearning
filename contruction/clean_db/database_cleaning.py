import os
from datetime import datetime

import pandas as pd
import pymongo
from dateutil.relativedelta import relativedelta


def get_out_of_caution(collection, dt):

    result = collection.aggregate([
        {
            '$match': {
                'timestamp':{"$lt": dt}
                # 'timestamp': same_time
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
        del df['_id']
    return df

def save_log(posts: pd.DataFrame, path: str) -> None:
    posts.to_csv(path, index=False)

def data_cleaning_routine(database_uri, database_name, collection_name, logpath="/home/damtien440/danaflood-flood-detection-using-camera-and-deeplearning/contruction/clean_db", caring_days=4):
    client = pymongo.MongoClient(database_uri)

    db = client[database_name]
    collection = db[collection_name]

    dt = datetime.now() - relativedelta(days=caring_days)
    to_be_delete = get_out_of_caution(collection, dt)
    if len(to_be_delete) > 0:
        collection.delete_many(
            {
                'timestamp':
                    # same_time
                    {  "$lt": dt}
        })
        collection.insert_many(to_be_delete.to_dict('records'))

        curr_timestamp = int(datetime.timestamp(datetime.now()))

        # save_log(to_be_delete,  logpath + os.sep + str(curr_timestamp)+'.csv')
        # print('saved log at', logpath + os.sep + str(curr_timestamp)+'.csv')

database_uri = "mongodb+srv://FLOODING_PROTOTYPE:FLOODING@cluster0.v1qjbym.mongodb.net/?retryWrites=true&w=majority"
database_name = 'danang_flood'
collection_name = 'camera_logging'
log_path = '/home/damtien440/danaflood-flood-detection-using-camera-and-deeplearning/contruction/clean_db'
if __name__ == '__main__':
    data_cleaning_routine(database_uri, database_name, collection_name, log_path, 4)

