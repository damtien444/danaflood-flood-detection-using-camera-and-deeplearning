import glob
import os
import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import cv2
import base64
import pymongo as pymongo

from helper import get_all_log, get_latest_unique_warning, get_last_record

client = pymongo.MongoClient(
    "mongodb+srv://FLOODING_PROTOTYPE:FLOODING@cluster0.v1qjbym.mongodb.net/?retryWrites=true&w=majority")
db = client['danang_flood']
collection = db['camera_logging']

st.set_page_config(
    page_title="Real-Time Flood Dashboard",
    # page_icon="✅",
    layout="wide",
)

# read csv from a URL

count = st_autorefresh(interval=60000, key="fizzbuzzcounter")


def get_data() -> pd.DataFrame:
    return get_all_log(collection)


def read_current_alert_cam_status(collection):
    current = get_latest_unique_warning(collection)
    return current


def get_latest_file_path(folder):
    images_folder = folder + os.sep + "*"
    list_of_files = glob.glob(images_folder)  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


df = get_data()

st.title("Real-Time Flood Dashboard")

st.write("Last update: " + str(df['timestamp'].max()))

kpi1, kpi2, alert_list = st.columns(3)

kpi1.metric(
    label='Number of Camera',
    value=str(len(pd.unique(df["name"])))
)

kpi1.dataframe(
    pd.unique(df["name"]),
)

current_alert_cam_status = read_current_alert_cam_status(collection)

try:
    number_of_warning = len(pd.unique(current_alert_cam_status["name"]))
except:
    number_of_warning = 0

kpi2.metric(
    label='Number of Alert',
    value=number_of_warning,
)

with alert_list:
    st.write("### Alert details..")
    if number_of_warning <= 0:
        st.write("There is no flooded event recorded.")
    else:
        st.dataframe(current_alert_cam_status)

st.write("## Sofi index of cameras")

df.sort_values(by=['timestamp'], inplace=True)

fig2 = px.line(data_frame=df, x="timestamp", y='sofi', color='name', markers=True, title="Static observer flood index")
fig2.update_yaxes(range=[0,1])
st.write(fig2)

selection = st.selectbox("Select camsite to see detail predictions", pd.unique(df['name']))

selected_infor = df[df['name'] == selection]
selected_infor.sort_values(by=['timestamp'], inplace=True,ascending=False)

image_place, data_place = st.columns(2)

last_selected_record = get_last_record(selection, collection)

with image_place:
    st.image(last_selected_record['image_b64'], caption=f"Latest image of {selection}.")

with data_place:
    st.dataframe(selected_infor)
