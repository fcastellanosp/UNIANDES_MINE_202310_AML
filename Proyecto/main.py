import Definitions

import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
# import plotly.graph_objects as go
import seaborn as sns

from datetime import datetime

from Labels import Labels

#st.title(" 📰 Pronóstico de temperatura del aire a 2 metros")
st.set_page_config(page_icon="🌡️", layout="wide", initial_sidebar_state="expanded")

lbl = Labels()

with st.form(key='FilterForm'):
    with st.sidebar:
        subtitle_1 = st.sidebar.title("Parámetros")
        subtitle_2 = st.sidebar.write("Para la estación")
        station_l1_sel = st.selectbox("Departamento", [], help="Estación de monitoreo")
        station_l2_sel = st.selectbox("Municipio", [], help="Estación de monitoreo")
        station_l3_sel = st.selectbox("Estación", [], help="Estación de monitoreo")
        subtitle_3 = st.sidebar.write("Para la temporalidad")
        initial_date = st.date_input("Fecha", lbl.default_init_date)

        station_sel = st.selectbox("Estación", ('Email', 'Home phone', 'Mobile phone'), help="Estación de monitoreo")

        submitted1 = st.form_submit_button(label='Submit 🔎')

    row_01_col1, row_01_col2 = st.columns(2)
    with row_01_col1:
        ":world_map: Visualización Geográfica :world_map:"
        df = pd.DataFrame(np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4], columns=['lat', 'lon'])
        st.map(df)

    if submitted1:
        a = ""

        with st.spinner("Procesando datos...."):
            st.caption("A continuación los resultados...", unsafe_allow_html=False)

            with row_01_col2:
                ":chart: Tendencia"

            row_02_col1, row_02_col2, row_02_col3 = st.columns([1, 2, 1])
            with row_02_col2:
                ":memo: Los datos insumo para el pronóstico"
                st.dataframe(df)

            row_03_col1, row_03_col2, row_03_col3 = st.columns([1, 2, 1])
            with row_03_col1:
                with st.expander("See source data"):
                    "Documentación del modelo"
