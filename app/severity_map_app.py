"""เว็บตัวอย่าง: คลิกจุดบนแผนที่ แล้วดูว่าถ้าเกิดอุบัติเหตุตรงนั้น มีโอกาสร้ายแรงกี่ %

วิธีรัน (จากโฟลเดอร์โปรเจกต์):
    uv run streamlit run app/severity_map_app.py

ต้องรัน notebooks/explore_idea/severity_map.ipynb ก่อน เพื่อสร้างไฟล์ข้อมูลและโมเดล
"""
from datetime import date
from pathlib import Path

import folium
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

PROJECT_DIR = Path(__file__).resolve().parent.parent
IDEA_DIR = PROJECT_DIR / "data" / "explore_idea" / "severity_map"
CLEAN_PATH = IDEA_DIR / "clean" / "accidents_clean.csv"
MODEL_PATH = IDEA_DIR / "model" / "severity_model.joblib"

# เกณฑ์ระดับความเสี่ยง ใช้ค่าเดียวกับใน notebook (หัวข้อ 10.2)
HIGH_RISK = 0.30
MEDIUM_RISK = 0.15

# ค่าเริ่มต้นของหน้าเว็บ
DEFAULT_PROVINCE = "จันทบุรี"
MAIN_VEHICLES = ["รถยนต์นั่ง", "รถกระบะ", "รถจักรยานยนต์", "รถตู้",
                 "รถบรรทุก 6 ล้อ", "รถบรรทุก 10 ล้อ", "รถบรรทุกมากกว่า 10 ล้อ"]
FEATURES = ["vehicle", "festival", "road_type", "hour", "day_of_week", "month", "lat", "lon"]
KM_PER_DEGREE = 111.0


@st.cache_data
def load_accidents():
    return pd.read_csv(CLEAN_PATH, encoding="utf-8-sig", parse_dates=["date"])


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def find_festival(month, day):
    # ต้องตรงกับฟังก์ชันชื่อเดียวกันใน notebook
    if month == 12 and day >= 27:
        return "ปีใหม่"
    if month == 1 and day <= 4:
        return "ปีใหม่"
    if month == 4 and day >= 10 and day <= 17:
        return "สงกรานต์"
    return "ปกติ"


def distance_km(table, lat, lon):
    # ระยะทางโดยประมาณ แม่นพอสำหรับรัศมีไม่กี่กิโลเมตร
    lat_km = (table["lat"] - lat) * KM_PER_DEGREE
    lon_km = (table["lon"] - lon) * KM_PER_DEGREE * np.cos(np.radians(lat))
    return np.sqrt(lat_km ** 2 + lon_km ** 2)


def make_input_row(lat, lon, chosen_date, hour, vehicle, road_type):
    return {
        "vehicle": vehicle,
        "festival": find_festival(chosen_date.month, chosen_date.day),
        "road_type": road_type,
        "hour": hour,
        "day_of_week": chosen_date.weekday(),
        "month": chosen_date.month,
        "lat": lat,
        "lon": lon,
    }


def predict(model, rows):
    table = pd.DataFrame(rows)
    return model.predict_proba(table[FEATURES])[:, 1]


def risk_label(probability):
    if probability >= HIGH_RISK:
        return "เสี่ยงสูง"
    if probability >= MEDIUM_RISK:
        return "เสี่ยงปานกลาง"
    return "เสี่ยงต่ำ"


def build_map(province_rows):
    center = [province_rows["lat"].median(), province_rows["lon"].median()]
    province_map = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap")

    # สีร้อน = เกิดเหตุบ่อย
    points = province_rows[["lat", "lon"]].values.tolist()
    HeatMap(points, radius=9, blur=12, min_opacity=0.3).add_to(province_map)

    # จุดแดง = เหตุร้ายแรง
    severe_rows = province_rows.loc[province_rows["severe"].eq(1)]
    for row in severe_rows.itertuples():
        folium.CircleMarker([row.lat, row.lon], radius=2, color="#c0392b",
                            fill=True, fill_opacity=0.8, weight=0).add_to(province_map)
    return province_map


st.set_page_config(page_title="Severity Map", layout="wide")
st.title("ถ้าเกิดอุบัติเหตุตรงนี้ จะร้ายแรงไหม")
st.caption("ต้นแบบสำหรับวางแผนจุดจอดรถกู้ชีพและจุดตรวจ | ข้อมูลอุบัติเหตุบนทางหลวงปี 2020–ต.ค. 2026")

if not CLEAN_PATH.exists() or not MODEL_PATH.exists():
    st.error("ไม่พบไฟล์ข้อมูลหรือโมเดล ให้รัน notebooks/explore_idea/severity_map.ipynb ก่อน")
    st.stop()

all_accidents = load_accidents()
model = load_model()
overall_rate = all_accidents["severe"].mean()
year_count = all_accidents["year"].nunique()

# แผนที่และสถิติรอบจุดใช้เฉพาะเหตุที่มีพิกัด
accidents = all_accidents.dropna(subset=["lat", "lon"])

# ---------- ช่องกรอกด้านซ้าย ----------
provinces = sorted(accidents["province"].unique().tolist())
if "ไม่ระบุ" in provinces:
    provinces.remove("ไม่ระบุ")

st.sidebar.header("1. เลือกพื้นที่")
province = st.sidebar.selectbox("จังหวัด", provinces, index=provinces.index(DEFAULT_PROVINCE))
radius_km = st.sidebar.slider("รัศมีที่ใช้ดูเหตุในอดีต (กม.)", 1, 10, 3)

st.sidebar.header("2. เลือกสถานการณ์")
vehicle = st.sidebar.selectbox("ประเภทรถ", MAIN_VEHICLES, index=MAIN_VEHICLES.index("รถจักรยานยนต์"))
chosen_date = st.sidebar.date_input("วันที่", value=date.today())
hour = st.sidebar.slider("เวลา (ชั่วโมง)", 0, 23, 20)

# เปลี่ยนจังหวัดแล้วให้ล้างจุดที่เคยคลิก
if st.session_state.get("province") != province:
    st.session_state["province"] = province
    st.session_state["clicked"] = None
clicked = st.session_state.get("clicked")

province_rows = accidents.loc[accidents["province"].eq(province)]

map_column, result_column = st.columns([3, 2])

# ---------- แผนที่ ----------
with map_column:
    st.subheader("3. คลิกจุดบนแผนที่")
    st.caption("สีร้อนคือบริเวณที่เกิดเหตุบ่อย จุดแดงคือเหตุร้ายแรงในอดีต")

    province_map = build_map(province_rows)
    selected_layer = folium.FeatureGroup(name="จุดที่เลือก")
    if clicked is not None:
        folium.Marker(clicked).add_to(selected_layer)
        folium.Circle(clicked, radius=radius_km * 1000, color="#1f4e79", fill=False).add_to(selected_layer)

    map_result = st_folium(province_map, feature_group_to_add=selected_layer, key="map_" + province,
                           height=560, use_container_width=True, returned_objects=["last_clicked"])

    if map_result is not None and map_result.get("last_clicked") is not None:
        new_point = [map_result["last_clicked"]["lat"], map_result["last_clicked"]["lng"]]
        if new_point != clicked:
            st.session_state["clicked"] = new_point
            st.rerun()

# ---------- ผลลัพธ์ ----------
with result_column:
    st.subheader("ผลลัพธ์")

    if clicked is None:
        st.info("คลิกจุดบนถนนในแผนที่ เพื่อดูสถิติย้อนหลังและโอกาสร้ายแรง")
        st.stop()

    lat = clicked[0]
    lon = clicked[1]

    # --- ส่วนที่ 1: สถิติย้อนหลัง (นับจากข้อมูลจริง ไม่ใช้โมเดล) ---
    nearby = accidents.loc[distance_km(accidents, lat, lon) <= radius_km]
    nearby_count = len(nearby)

    st.markdown("**เกิดบ่อยไหม** (นับจากเหตุในอดีต รัศมี " + str(radius_km) + " กม.)")
    count_column, severe_column = st.columns(2)
    count_column.metric("จำนวนเหตุ", str(nearby_count) + " ครั้ง")
    count_column.caption("เฉลี่ย " + str(round(nearby_count / year_count, 1)) + " ครั้ง/ปี")
    if nearby_count == 0:
        # โมเดลไม่เคยเห็นข้อมูลแถวนี้ จึงไม่แสดงผลทำนาย
        severe_column.metric("เหตุร้ายแรงในอดีต", "0 ครั้ง")
        st.warning("ไม่มีเหตุในอดีตในรัศมีนี้ จุดนี้อาจไม่ได้อยู่บนทางหลวง จึงไม่แสดงผลทำนาย "
                   "ลองคลิกบนถนนที่มีสีร้อน หรือเพิ่มรัศมี")
        st.stop()

    severe_column.metric("เหตุร้ายแรงในอดีต", str(int(nearby["severe"].sum())) + " ครั้ง")
    severe_column.caption(str(round(nearby["severe"].mean() * 100, 1)) + "% ของเหตุในรัศมีนี้")
    # ประเภทถนนดูจากเหตุในอดีตรอบจุดที่คลิก ผู้ใช้ไม่ต้องกรอก
    road_type = nearby["road_type"].value_counts().index[0]

    # --- ส่วนที่ 2: โอกาสร้ายแรงจากโมเดล ---
    probability = predict(model, [make_input_row(lat, lon, chosen_date, hour, vehicle, road_type)])[0]

    st.markdown("**ถ้าเกิดเหตุแล้ว ร้ายแรงไหม** (จากโมเดล)")
    st.metric("โอกาสร้ายแรง", str(round(probability * 100, 1)) + "%")
    st.markdown("**" + risk_label(probability) + "** (ค่าเฉลี่ยทั้งประเทศ "
                + str(round(overall_rate * 100, 1)) + "%)")
    st.caption(vehicle + " | " + chosen_date.strftime("%d/%m/%Y") + " เวลา " + str(hour) + ":00 น. | "
               + road_type + " | ช่วง" + find_festival(chosen_date.month, chosen_date.day))

    # --- ส่วนที่ 3: เทียบประเภทรถ ในจุดและเวลาเดียวกัน ---
    vehicle_rows = []
    for name in MAIN_VEHICLES:
        vehicle_rows.append(make_input_row(lat, lon, chosen_date, hour, name, road_type))
    vehicle_chart = pd.DataFrame({"โอกาสร้ายแรง (%)": predict(model, vehicle_rows) * 100}, index=MAIN_VEHICLES)
    st.markdown("**เทียบประเภทรถ** (จุดและเวลาเดียวกัน)")
    st.bar_chart(vehicle_chart, horizontal=True, height=260)

    # --- ส่วนที่ 4: เทียบเวลา สำหรับรถประเภทที่เลือก ---
    hour_rows = []
    for each_hour in range(24):
        hour_rows.append(make_input_row(lat, lon, chosen_date, each_hour, vehicle, road_type))
    hour_chart = pd.DataFrame({"โอกาสร้ายแรง (%)": predict(model, hour_rows) * 100}, index=range(24))
    hour_chart.index.name = "ชั่วโมง"
    st.markdown("**เทียบเวลา** (" + vehicle + ")")
    st.line_chart(hour_chart, height=220)

    # --- ส่วนที่ 5: ลักษณะเหตุในอดีตรอบจุดนี้ ---
    st.markdown("**เหตุในอดีตรอบจุดนี้เป็นแบบไหน**")
    crash_table = nearby["crash_type"].value_counts().head(5).rename("จำนวนเหตุ").to_frame()
    crash_table.index.name = "ลักษณะการชน"
    st.dataframe(crash_table, width="stretch")

    st.caption("ตัวเลขนี้คือโอกาสร้ายแรงในกลุ่มเหตุที่ถูกบันทึก ไม่ใช่โอกาสที่จะเกิดอุบัติเหตุ "
               "และเป็นความสัมพันธ์จากข้อมูลในอดีต ไม่ได้พิสูจน์สาเหตุ")
