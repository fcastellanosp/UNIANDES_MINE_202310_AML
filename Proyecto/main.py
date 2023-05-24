import Definitions
import streamlit as st
import matplotlib.pyplot as plt
import os.path as osp
import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime

from Labels import Labels
from back.DataController import DataController

# Variables
min_instances = 150
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
        st.caption("Mapa de estaciones en Colombia...", unsafe_allow_html=False)

    # "Datos de las estaciones"
    with row_01_col2:
        ":memo: Los datos de estaciones"
        st.dataframe(stations_df[["codigo", "departamento", "municipio", "nombre", "categoria", "altitud"]])
        st.metric(label="Total", value=stations_df.shape[0], delta=None)

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

        initial_date_sel = initial_date.strftime("%Y-%m-%d")
        ending_date_sel = ending_date.strftime("%Y-%m-%d")
        hour_sel = hour.hour
        # hour_sel = 12
        # print(str(station_l3_sel))

        station_sel = stations_df[(stations_df['nombre'] == str(station_l3_sel))]["codigo"].item()
        # station_sel = station_sel[-8:]
        # print(station_sel)

        # station_sel = "0021205012"
        # station_sel = "0052055170"
        # hour_sel = 1
        # print(station_sel)

        input_data_ok = True
        date_diff = ending_date - initial_date
        if date_diff.days < 90:
            input_data_ok = False

        model_name = f"{station_sel}_h{hour}"
        if not osp.join(Definitions.ROOT_DIR, "resources/models", f"{model_name}.h5"):
            input_data_ok = False

        print(f"input_data_ok: {input_data_ok}")

        if input_data_ok:
            with st.spinner("Procesando datos...."):

                # PENDIENTE POR AJUSTAR
                # with row_01_col1:
                    # "El mapa"

                ## "Datos de las estaciones"
                #with row_01_col2:
                #    ":memo: Los datos de estaciones"
                #    st.dataframe(stations_df[["codigo", "departamento", "municipio", "nombre", "categoria", "altitud"]])
                #    st.metric(label="Total", value=stations_df.shape[0], delta=None)

                # "Datos de las temperaturas"
                row_02_col1, row_02_col2 = st.columns(2)
                with row_02_col1:
                    ":memo: Datos de las observaciones"
                    temp_df = controller.query_temp_station_values(station_sel, initial_date_sel, ending_date_sel)
                    # print(temp_df)
                    if temp_df.shape[0] > 0:
                        temp_df = temp_df[temp_df["hora"] == hour_sel]
                        st.dataframe(temp_df.sort_values(by='fecha'))
                        st.metric(label="Total", value=temp_df.shape[0], delta=None)
                        st.caption("Es posible que no existan datos de todas las fechas", unsafe_allow_html=False)
                        if temp_df.shape[0] < min_instances:
                            st.caption(f"Se requieren al menos ({min_instances}) registros, cuenta con: {temp_df.shape[0]}", unsafe_allow_html=False)

                with row_02_col2:
                    ":chart_with_upwards_trend: Comportamiento de las observaciones"
                    if temp_df.shape[0] > 0:
                        temp_fig = px.line(temp_df, x="fecha", y="observacion", title='Temperaturas')
                        st.plotly_chart(temp_fig)

                # "Datos de las estimaciones"
                if temp_df.shape[0] > 0 and temp_df.shape[0] >= min_instances:
                    title, model_dates, X_Real_val, trainPredictPlot, testPredictPlot, metrics = \
                        controller.predict(temp_df, station_code=station_sel, hour=hour_sel)

                pred_error_msg = "No fue posible generar la predicción con los parámetros indicados"
                # row_03_col1, row_03_col2 = st.columns([4, 2])
                row_03_col1, row_03_col2 = st.columns(2)
                with row_03_col1:
                    try:
                        pred_fig = go.Figure()
                        pred_fig.add_trace(go.Scatter(x=model_dates, y=X_Real_val, mode='lines+markers', name='Real'))
                        pred_fig.add_trace(go.Scatter(x=model_dates, y=trainPredictPlot, mode='lines+markers', name='Estimado'))
                        pred_fig.add_trace(go.Scatter(x=model_dates, y=testPredictPlot, mode='lines+markers', name='Predecido'))
                        pred_fig.update_layout(title=title, xaxis_title='Dia', yaxis_title='Temperatura')

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
                        "Los modelos LSTM utilizados en este proyecto han demostrado ser eficaces para analizar la variación temporal de la temperatura mínima en Colombia. Estos modelos tienen la capacidad de capturar patrones complejos en los datos y generar predicciones precisas. Esto sugiere que los modelos LSTM pueden ser una herramienta útil para estudiar y predecir el clima en otras regiones o para analizar diferentes variables climáticas."
                        "Los resultados obtenidos en este proyecto resaltan la importancia de monitorear y comprender los cambios en la temperatura mínima en Colombia. Estos cambios pueden tener impactos significativos en diferentes aspectos, como la agricultura, la salud pública y los ecosistemas. El análisis de la variación temporal de la temperatura mínima proporciona información valiosa para comprender mejor el clima y tomar decisiones informadas en áreas relacionadas."
                        ""
                        f"© Datos de estaciones de IDEAM compartidos en {controller.open_data_host}"
                        st.write(
                            "Código fuente [link](https://github.com/fcastellanosp/UNIANDES_MINE_202310_AML/tree/main/Proyecto)")


