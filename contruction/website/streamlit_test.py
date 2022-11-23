import glob
import os
import time  # to simulate a real time data, time loop

import numpy as np  # np mean, np random
import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import streamlit as st  # 🎈 data web app development
from streamlit_autorefresh import st_autorefresh

st.set_page_config(
    page_title="Real-Time Data Science Dashboard",
    # page_icon="✅",
    layout="wide",
)

cvs_path = r"E:\DATN_local\2_HISTORY_INFERENCE\mobilenet_v2_2022-11-22.json"
image_path = r"E:\DATN_local\2_HISTORY_INFERENCE\2022-11-22"
names = ["CAMERA_SITE", "Timestamp", "Flood Wanring Level", "SOFI index"]
# read csv from a URL

count = st_autorefresh(interval=2000, key="fizzbuzzcounter")


def get_data() -> pd.DataFrame:
    return pd.read_csv(cvs_path, names=names)


def read_current_cam_status(df):
    current = df.groupby(['CAMERA_SITE'])['Timestamp'].transform(max) == df['Timestamp']
    return df[current]


def number_cam_allert(status):
    return status[df['Flood Wanring Level'] > 1]

def get_latest_file_path(folder):
    images_folder = folder + os.sep + "*"
    list_of_files = glob.glob(images_folder) # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file


df = get_data()

st.title("Real-Time Flood Dashboard")

kpi1, kpi2, alert_list = st.columns(3)

kpi1.metric(
    label='Number of Camera',
    value=len(pd.unique(df["CAMERA_SITE"]))
)

current_cam_status = read_current_cam_status(df)

list_of_alert = number_cam_allert(current_cam_status)

kpi2.metric(
    label='Number of Alert Site',
    value=len(list_of_alert),
)

with alert_list:
    st.write("### Alert details..")
    st.dataframe(list_of_alert)

st.write("## Sofi index of cameras")
fig2 = px.line(data_frame=df, x="Timestamp", y='SOFI index', color='CAMERA_SITE', markers=True, title="S")
st.write(fig2)

selection = st.selectbox("Select camsite to see detail predictions", pd.unique(df['CAMERA_SITE']))

selected_infor = df[df['CAMERA_SITE']== selection]

image_place, data_place = st.columns(2)

image_folder_path = image_path + os.sep + selection

latest_image_path = get_latest_file_path(image_folder_path)

from PIL import Image
image = Image.open(latest_image_path)

with image_place:
    st.image(image, caption=f"Latest image of {selection}.")

with data_place:
    st.dataframe(selected_infor)
