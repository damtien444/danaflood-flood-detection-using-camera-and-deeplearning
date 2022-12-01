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

st.markdown('<div style="text-align: justify;">Extreme weather occurs more frequently and has a negative impact on '
            'urban areas, especially those in developing countries. Nearly every city has access to surveillance '
            'camera systems, but they lack a smart function that would send an alert in an emergency. As a result, '
            'we propose a highly scalable intelligent system for alarming street flooding. This system is capable of '
            'simultaneously producing high-resolution data for future use and sending out high-abstract warning '
            'signals. The chosen deep convolutional neural network model, MobileNetV2, achieved classification '
            'accuracy of 89.58% and flood image segmentation accuracy of 95.33%.</div>', unsafe_allow_html=True)
st.write("\n")
st.markdown('<div style="text-align: justify;">The developed methodology is provided in <a href="https://dutudn-my.sharepoint.com/:b:/g/personal/102180048_sv1_dut_udn_vn/ESrrWy3LG05EkVv-rJCeKuEB4naemi7RWcyYBjmrPNIUDg?e=SnpPnz">this paper.</a></div>', unsafe_allow_html=True)

st.write("")
st.write("Last update: " + str(df['timestamp'].max()))

kpi1, kpi2= st.columns(2)
kpi1.write("### Camera information")
kpi1.markdown(f'<div style="text-align: justify;">Because of privacy concerns and security camera accessing policies, '
              f'this demo page can only provide you with <b>{len(pd.unique(df["name"]))} camera site(s)</b> in '
              f'Danang, as listed in the dataframe below. The author thanks <a href="https://camera.0511.vn/camera.html">Hội Phát Triển Sáng Tạo Đà Nẵng</a> for granting permission to use the cameras.</div>', unsafe_allow_html=True)

kpi1.write("\n")
kpi1.dataframe(
    pd.unique(df["name"]),
)

current_alert_cam_status = read_current_alert_cam_status(collection)

try:
    number_of_warning = len(pd.unique(current_alert_cam_status["name"]))
except:
    number_of_warning = 0

kpi2.write("### Alert notation")

kpi2.markdown('<div style="text-align: justify;">This model can generate high-abstract warning signals as well as high-resolution logging data that can be used for a variety of purposes. They are:</div>', unsafe_allow_html=True)

kpi2.write("#### 1. Warning Index")
kpi2.markdown('<div style="text-align: justify;">This is the output of the classification head, which categorizes scenes as (0) no water, (1) water but not affected, (2) not recommend to commute, and (3) dangerous to commute.</div>', unsafe_allow_html=True)
kpi2.write("\n")
kpi2.markdown(f'<div style="text-align: justify;">In the last 24 hours, there has been <b>{number_of_warning} camera site(s)</b> having warning index greater than 1.</div>', unsafe_allow_html=True)

kpi2.write("#### 2. Static Observer Flooding Index")
kpi2.markdown('<div style="text-align: justify;">The static observer flooding index (SOFI) is introduced as a dimensionless proxy for water level fluctuation that can be extracted from segmented images of stationary surveillance cameras. It\'s computed as the below formula:</div>', unsafe_allow_html=True)
kpi2.write("\n")
kpi2.markdown(r'''$$SOFI=\frac{\#flooding_{pixels}}{\#total_{pixel}}$$''')

st.write("## Historical SOFIs")

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
