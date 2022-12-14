import datetime

import pytz

import pandas as pd  # read csv, df manipulation
import plotly.express as px  # interactive charts
import pymongo as pymongo
import streamlit as st  # 🎈 data web app development
from dateutil.relativedelta import relativedelta
from streamlit_autorefresh import st_autorefresh
from streamlit_plotly_events import plotly_events

from helper import get_all_log, get_latest_unique_warning, get_last_record, get_a_record

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
now =  datetime.datetime.utcnow().astimezone(pytz.timezone('Asia/Ho_Chi_Minh'))
now = now.replace(tzinfo=None)
def get_data() -> pd.DataFrame:
    return get_all_log(now, collection)


def read_current_alert_cam_status(collection):
    current = get_latest_unique_warning(collection)
    return current


def get_current_cam_status(df):
    current = df.groupby(['name'])['timestamp'].transform(max) == df['timestamp']
    return df[current]


def number_cam_alert(status):
    return status[df['warning_index']>1]


df = get_data()
current = get_current_cam_status(df)
alert = number_cam_alert(current)

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
              f'Danang, as listed in the dataframe below. The author thanks <a '
              f'href="https://camera.0511.vn/camera.html">Hội Phát Triển Sáng Tạo Đà Nẵng</a> for granting '
              f'permission to use the cameras.</div>', unsafe_allow_html=True)

kpi1.write("\n")
kpi1.dataframe(
    current,
)

try:
    number_of_warning = len(alert)
except:
    number_of_warning = 0
kpi1.markdown(f'<div style="text-align: justify;">Curently, there are <b>{number_of_warning} camera site(s)</b> being clasified as not recommend or not safety.</div>', unsafe_allow_html=True)



kpi2.write("### Alert notation")

kpi2.markdown('<div style="text-align: justify;">This model can generate high-abstract warning signals as well as high-resolution logging data that can be used for a variety of purposes. They are:</div>', unsafe_allow_html=True)

kpi2.write("#### 1. Warning Index")
kpi2.markdown('<div style="text-align: justify;">This is the output of the classification head, which categorizes scenes as (0) no water, (1) water but not affected, (2) not recommend to commute, and (3) dangerous to commute.</div>', unsafe_allow_html=True)
kpi2.write("\n")


kpi2.write("#### 2. Static Observer Flooding Index")
kpi2.markdown('<div style="text-align: justify;">The static observer flooding index (SOFI) is introduced as a dimensionless proxy for water level fluctuation that can be extracted from segmented images of stationary surveillance cameras. It\'s computed as the below formula:</div>', unsafe_allow_html=True)
kpi2.write("\n")
kpi2.markdown(r'''$$SOFI=\frac{\#flooding_{pixels}}{\#total_{pixel}}$$''')

st.write("### Current cam status")
st.write("Inference results come in two forms, images and dataframes. On inferenced images, parts having bright white "
         "color are segmented area clasified as flooded. The dataframe is quite straight forward with each record "
         "containing name, timestamp, warning index and sofi. The dataframe is ordered backward in time.")
selection = st.selectbox("Select camsite to see prediction detail", pd.unique(df['name']))

selected_infor = df[df['name'] == selection]
selected_infor.sort_values(by=['timestamp'], inplace=True,ascending=False)

image_place, data_place = st.columns(2)

last_selected_record = get_last_record(selection, collection)

with image_place:
    st.image(last_selected_record['image_b64'], caption=f"Latest image of {selection}.")

with data_place:
    st.dataframe(selected_infor)


st.write("## Historical timeline")

# st.selectbox()
st.write("")
date_range_choice = st.radio("Select date range to see the progress of street flood in",['Last 24 hours', 'Last 3 days'])
if date_range_choice == 'Last 24 hours':
    start_date = now - relativedelta(days=1)
    start_care_df = df['timestamp'].searchsorted(start_date)
    end_care_df = df['timestamp'].searchsorted(now)
    ranged_df = df.loc[start_care_df:end_care_df - 1]

elif date_range_choice == 'Last 3 days':
    start_date = now - relativedelta(days=3)
    start_care_df = df['timestamp'].searchsorted(start_date)
    end_care_df = df['timestamp'].searchsorted(now)
    ranged_df = df.loc[start_care_df:end_care_df - 1]
st.write("There was an importance database and model update on 2022/12/07, so that you may observe a big data gap. The camera "
         "'PHUONGTRAN' had low quality night-time images, we had to ignore that camera, but we still keep its "
         "previous data for reference purposes. Due to the lack of resources, our system can only store data in "
         "three-day period.")
st.write("Tips: When you click any data point, the screen will reset and provide the image and inference details of "
         "that camera site.")

ranged_df.sort_values(by=['timestamp'], inplace=True)

fig1 = px.line(data_frame=ranged_df, x="timestamp", y='sofi', color='name', markers=True, title="Static observer flood index", template="seaborn")
fig1.update_yaxes(range=[0,1])
selected_points = plotly_events(fig1)

if selected_points:
    st.write("Detail of selected datapoint")
    image_col, data_col = st.columns(2)
    print(selected_points)
    selected_record = ranged_df.loc[(ranged_df['timestamp']==selected_points[0]['x']) & (ranged_df['sofi'] == selected_points[0]['y'])]
    selected_record = selected_record.iloc[0]
    data_col.write(selected_record)
    image_col.image(get_a_record(selected_record['name'], selected_record['timestamp'], collection)['image_b64'])
    selected_points = None

# st.write(fig1)

fig2 = px.line(data_frame=ranged_df, x="timestamp", y='warning_index', color='name', markers=True, title="Warning index", template="seaborn")
fig2.update_yaxes(range=[0,3])
fig2.update_layout()
selected_points = plotly_events(fig2)

if selected_points:
    st.write("Detail of selected datapoint")
    image_col, data_col = st.columns(2)
    print(selected_points)
    selected_record = ranged_df.loc[(ranged_df['timestamp']==selected_points[0]['x']) & (ranged_df['warning_index'] == selected_points[0]['y'])]
    selected_record = selected_record.iloc[0]
    data_col.write(selected_record)
    image_col.image(get_a_record(selected_record['name'], selected_record['timestamp'], collection)['image_b64'])
    selected_points = None

# st.write(fig2)



