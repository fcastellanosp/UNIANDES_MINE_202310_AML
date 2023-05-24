import Definitions

import altair as alt
import json
import geopandas as gpd
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

from datetime import datetime

from Labels import Labels
from back.DataController import DataController

# Variables
lbl = Labels()
controller = DataController(True)
stations_df = controller.stations_df
hour_dt = datetime.now()
hour_dt = hour_dt.replace(hour=12, minute=0, second=0)

# st.title(" 📰 {}".format(lbl.main_title))
st.set_page_config(page_icon="🌡️", layout="wide", initial_sidebar_state="expanded")

with st.form(key='FilterForm'):
    with st.sidebar:
        subtitle_1 = st.sidebar.title("Parámetros")
        subtitle_2 = st.sidebar.write("Para la estación")
        station_l1_sel = st.selectbox("Departamento", controller.query_dpto(), help="Estación de monitoreo")
        station_l2_sel = st.selectbox("Municipio", [], help="Estación de monitoreo")

        station_l3_sel = st.selectbox("Estación", stations_df[["nombre"]].sort_values(by='nombre'), help="Estación de monitoreo")

        subtitle_3 = st.sidebar.write("Para la temporalidad")
        initial_date = st.date_input("Fecha Inicial", lbl.default_init_date)
        ending_date = st.date_input("Fecha Final", lbl.default_ending_date)
        # subtitle_4 = st.sidebar.write("Para el pronóstivo")
        hour = st.time_input('Hora', value=hour_dt, step=60*60, help="Permitirá ver la información basado en la hora del día")

        submitted1 = st.form_submit_button(label='Generar 🔎')

    row_01_col1, row_01_col2 = st.columns(2)
    with row_01_col1:
        ":world_map: Visualización Geográfica :world_map:"
        # st.map(df)
        # print(stations_df[['lat', 'lon']])
        st.map(stations_df[['lat', 'lon']])

        # stations_gdf = gpd.GeoDataFrame(
        #     stations_df, geometry=gpd.points_from_xy(stations_df.lon, stations_df.lat), crs="EPSG:4326"
        # )
        #
        # fig = px.choropleth(stations_gdf,
        # # fig = px.choropleth_mapbox(stations_gdf,
        #                     geojson=stations_gdf.geometry,
        #                     color_continuous_scale="Viridis",
        #                     # zoom=10, center={"lat": 4, "lon": -74},
        #                     locations=stations_gdf.index,
        #                     color="altitud",
        #                     projection="mercator")
        # fig.update_geos(fitbounds="locations", visible=False)
        # st.plotly_chart(fig)

    if submitted1:
        with st.spinner("Procesando datos...."):
            st.caption("A continuación los resultados...", unsafe_allow_html=False)

            # PENDIENTE POR AJUSTAR
            # with row_01_col1:
                # "El mapa"

            # "Datos de las estaciones"
            with row_01_col2:
                ":memo: Los datos de estaciones"
                st.dataframe(stations_df[["codigo", "departamento", "municipio", "nombre", "categoria", "altitud"]])
                st.metric(label="Total", value=stations_df.shape[0], delta=None)

            # "Datos de las temperaturas"
            row_02_col1, row_02_col2 = st.columns(2)
            with row_02_col1:
                ":memo: Datos de las observaciones"
                temp_df = controller.query_temp_station_values()
                st.dataframe(temp_df)
                st.metric(label="Total", value=temp_df.shape[0], delta=None)

            with row_02_col2:
                ":chart_with_upwards_trend: Comportamiento de las observaciones"
                temp_fig = px.line(temp_df, x="fecha", y="observacion", title='Temperaturas')
                st.plotly_chart(temp_fig)

            # "Datos de las estimaciones"
            title, model_dates, X_Real_val, trainPredictPlot, testPredictPlot, metrics = \
                controller.predict(temp_df, station_code="0021205012", hour=12)

            pred_error_msg = "No fue posible generar la predicción con los parámetros indicados"
            # row_03_col1, row_03_col2 = st.columns([4, 2])
            row_03_col1, row_03_col2 = st.columns(2)
            with row_03_col1:
                try:
                    pred_fig = go.Figure()
                    pred_fig.add_trace(go.Scatter(x=model_dates, y=X_Real_val, mode='lines+markers', name='Real'))
                    pred_fig.add_trace(go.Scatter(x=model_dates, y=trainPredictPlot, mode='lines+markers', name='Estimado'))
                    pred_fig.add_trace(go.Scatter(x=model_dates, y=testPredictPlot, mode='lines+markers', name='Predecido'))

                    pred_fig.update_layout(title=title, xaxis_title='Dia', yaxis_title='Cantidad')

                    st.plotly_chart(pred_fig)
                except:
                    pred_error_msg

            with row_03_col2:
                ":chart_with_upwards_trend: Métricas"
                try:
                    st.dataframe(metrics)
                except:
                    pred_error_msg

            row_04_col1, row_04_col2 = st.columns([4, 1])

            row_04_col1, row_04_col2, row_04_col3 = st.columns([1, 2, 1])
            with row_04_col2:
                with st.expander("Mas Información"):
                    "Documentación del modelo"

