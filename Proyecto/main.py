import Definitions
import folium
import folium.plugins as fp
import os.path as osp
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from streamlit_folium import st_folium

from Parameters import Parameters
from back.DataController import DataController

# Variables
param = Parameters()
controller = DataController(True)
CENTER_START = [4, -74]
ZOOM_START = 4

# Estados
if "center" not in st.session_state:
    st.session_state["center"] = [4, -74]
if "markers" not in st.session_state:
    st.session_state["markers"] = []
if "zoom" not in st.session_state:
    st.session_state["zoom"] = 4
if "all_stations" not in st.session_state:
    st.session_state['all_stations'] = controller.stations_df
if "station_l1" not in st.session_state:
    st.session_state['station_l1'] = [param.default_selection] + controller.query_dep().values.tolist()
if "station_l2" not in st.session_state:
    st.session_state['station_l2'] = [param.default_selection]
if "station_l3" not in st.session_state:
    st.session_state['station_l3'] = [param.default_selection]
if "current_stations" not in st.session_state:
    st.session_state['current_stations'] = controller.stations_df[param.columns_sel]

main_map = folium.Map(location=CENTER_START, zoom_start=ZOOM_START)

# st.title(" 📰 {}".format(lbl.main_title))
st.set_page_config(page_icon="🌡️", layout="wide", initial_sidebar_state="expanded")


# Visualizar los valores asociados a los municipios
def station_l2_options(l1_sel):
    st.session_state['station_l3'] = [param.default_selection]
    if l1_sel != param.default_selection:
        res = controller.query_mun(l1_sel)
        return [param.default_selection] + res.values.tolist() if res.shape[0] > 0 else [param.default_selection]
    else:
        return [param.default_selection]


# Visualizar los valores asociados a las estaciones
def station_l3_options(l1_sel, l2_sel):
    st.session_state['station_l2'] = [l2_sel]
    if l1_sel != param.default_selection and l2_sel == param.default_selection:
        res = controller.query_stations_by_dep(l1_sel)
        return [param.default_selection] + res.values.tolist() if res.shape[0] > 0 else [param.default_selection]
    elif l2_sel != param.default_selection:
        res = controller.query_stations_by_mun(l2_sel)
        return [param.default_selection] + res.values.tolist() if res.shape[0] > 0 else [param.default_selection]
    return [param.default_selection]

def get_prediction(data):
    title, model_dates, X_Real_val, trainPredictPlot, \
        testPredictPlot, metrics = controller.predict(data, station_code=station_sel, hour=hour_sel)

    pred_fig = go.Figure()
    pred_fig.add_trace(go.Scatter(x=model_dates, y=X_Real_val, mode='lines+markers', name='Real'))
    pred_fig.add_trace(go.Scatter(x=model_dates, y=trainPredictPlot, mode='lines+markers', name='Estimado'))
    pred_fig.add_trace(go.Scatter(x=model_dates, y=testPredictPlot, mode='lines+markers', name='Predecido'))
    pred_fig.update_layout(title=title, xaxis_title='Dia', yaxis_title='Temperatura')

    return pred_fig, metrics

# Visualizar las estaciones actuales
def show_stations(l1_sel, l2_sel, l3_sel):
    print("station_l3_options ->")


# Presentación de resultados
row_01_col1, row_01_col2 = st.columns(2)
row_02_col1, row_02_col2 = st.columns(2)
row_03_col1, row_03_col2 = st.columns(2)
row_04_col1, row_04_col2, row_04_col3 = st.columns([1, 2, 1])

with st.form(key='FilterForm'):
    with st.sidebar:
        subtitle_1 = st.sidebar.title(param.form_title_lbl)
        subtitle_2 = st.sidebar.caption(param.form_subtitle1_lbl)
        station_l1_sel = st.sidebar.selectbox(label=param.station_l1_lbl, options=st.session_state['station_l1'],
                                              help=param.station_l1_tooltip, key="station_l1_id")
        station_l2_sel = st.sidebar.selectbox(label=param.station_l2_lbl, options=station_l2_options(station_l1_sel),
                                              help=param.station_l2_tooltip, key="station_l2_id")
        station_l3_sel = st.sidebar.selectbox(label=param.station_l3_lbl,
                                              options=station_l3_options(station_l1_sel, station_l2_sel),
                                              help=param.station_l3_tooltip, key="station_l3_id")

        subtitle_3 = st.sidebar.caption(param.form_subtitle2_lbl)
        initial_date = st.sidebar.date_input(label=param.init_date_lbl, value=param.init_date_val,
                                             help=param.form_init_tooltip)
        ending_date = st.sidebar.date_input(label=param.end_date_lbl, value=param.ending_date_val,
                                            help=param.form_end_tooltip)
        hour = st.sidebar.time_input(label='Hora', value=param.default_hour_val, step=param.hour_step,
                                     help=param.hour_tooltip)

        submitted = st.sidebar.button('Ver predicciones 🔎')

        # Botón en la barra lateral
        if submitted:
            # st.caption("Presentando datos...")

            initial_date_sel = initial_date.strftime("%Y-%m-%d")
            ending_date_sel = ending_date.strftime("%Y-%m-%d")
            hour_sel = hour.hour

            input_data_ok = True
            date_diff = ending_date - initial_date
            if date_diff.days < 90:
                input_data_ok = False

            if station_l3_sel != param.default_selection:
                stations_df = st.session_state['all_stations']
                station_sel = stations_df[(stations_df['nombre'] == str(station_l3_sel))]["codigo"].item()
                model_name = f"{station_sel}_h{hour_sel}"
                print(model_name)
                if not osp.join(Definitions.ROOT_DIR, "resources/models", f"{model_name}.h5"):
                    input_data_ok = False
            else:
                input_data_ok = False

            print(f"input_data_ok: {input_data_ok}")

            if input_data_ok:
                with st.spinner("Procesando datos...."):
                    # st.success('This is a success message!', icon="✅")
                    # st.info('This is a purely informational message', icon="ℹ️")

                    #with row_01_col1:
                        # st.session_state["center"] = [6, -76]
                        # st.session_state["zoom"] = 10
                        # st.session_state["markers"] = [6, -76]

                        # fg = folium.FeatureGroup(name="Estacion")
                        # for marker in st.session_state["markers"]:
                            # fg.add_child(marker)

                        # st_folium(
                        #     main_map,
                        #     center=st.session_state["center"],
                        #     zoom=st.session_state["zoom"],
                        #     # feature_group_to_add=fg,
                        #     key="updated_map"
                        # )

                    # Datos de las estaciones
                    with row_01_col2:
                        ce = controller.stations_df[param.columns_sel]
                        ce = ce[ce["codigo"] == station_sel]
                        st.session_state['current_stations'] = ce

                    # "Datos de las temperaturas"
                    with row_02_col1:
                        ":memo: Datos de las observaciones"
                        temperature_placeholder = st.empty()
                        temp_df = controller.query_temp_station_values(station_sel, initial_date_sel, ending_date_sel)
                        # print(temp_df)
                        if temp_df.shape[0] > 0:
                            temp_df = temp_df[temp_df["hora"] == hour_sel]
                            temperature_placeholder.dataframe(temp_df.sort_values(by='fecha'))
                            st.metric(label="Total", value=temp_df.shape[0], delta=None)
                            st.caption("Es posible que no existan datos de todas las fechas", unsafe_allow_html=False)
                            if temp_df.shape[0] < param.min_instances:
                                st.caption(
                                    f"Se requieren al menos ({param.min_instances}) registros, cuenta con: {temp_df.shape[0]}",
                                    unsafe_allow_html=False)

                    with row_02_col2:
                        ":chart_with_upwards_trend: Comportamiento de las observaciones"
                        if temp_df.shape[0] > 0:
                            temp_fig = px.line(temp_df, x="fecha", y="observacion", title='Temperaturas')
                            st.plotly_chart(temp_fig)

                    pred_error_msg = "No fue posible generar la predicción con los parámetros indicados"
                    with row_03_col1:
                        try:
                            # "Datos de las estimaciones"
                            if temp_df.shape[0] > 0 and temp_df.shape[0] >= param.min_instances:
                                pred_fig, metrics = get_prediction(temp_df)

                                st.plotly_chart(pred_fig)
                        except:
                            pred_error_msg

                    with row_03_col2:
                        ":chart_with_upwards_trend: Métricas"
                        try:
                            st.dataframe(metrics)
                        except:
                            pred_error_msg
            else:
                st.info('Parámetros incompletos', icon="ℹ")

    with row_01_col1:
        ":world_map: Visualización Geográfica :world_map:"
        st.caption("Estaciones para toma de datos en Colombia...")
        # map_placeholder = st.empty()
        # map_placeholder.map(controller.stations_df[['lat', 'lon']])

        heatmap_data = controller.stations_df[['lat', 'lon', 'altitud']].values.tolist()
        fp.HeatMap(heatmap_data, radius=15).add_to(main_map)

        # st_data = st_folium(
        #     main_map,
        #     center=st.session_state["center"],
        #     zoom=st.session_state["zoom"],
        #     key="initial_map",
        #     # feature_group_to_add=fg,
        #     height=350,
        #     width=700
        # )
        map_html = main_map._repr_html_()
        st.components.v1.html(map_html, width=700, height=350, scrolling=False)

    # Datos de las estaciones
    with row_01_col2:
        ":memo: Estacion(es) seleccionada(s)"
        stations_placeholder = st.empty()
        # df = controller.stations_df[param.columns_sel]
        stations_placeholder.dataframe(st.session_state['current_stations'])
        st.metric(label="Total", value=st.session_state['current_stations'].shape[0], delta=None)

    with row_02_col1:
        temperature_placeholder = st.empty()

    with row_04_col2:
        with st.expander("Mas Información"):
            f"""
            El aplicativo dispone la información de modelos entrenados para estaciones en Colombia en el periodo de tiempo
            anteriormente indicado, el mínimo de datos considerado es de {param.min_instances} instancias. 
            """

            st.info(
                f"""
                [Entrenamiento] Tomado de la información del año 2018 al año 2021 para los datos de temperatura mínima 
                del aire a 2 metros.
                © Datos de estaciones de IDEAM compartidos en {controller.open_data_host}
                """
            )

            """
            Los modelos fueron entrenados mediante redes recurrentes LSTM con una ventana de 7 días, permitiendo buscar
            sobre los datos distintas tendencias que pudiera presentar la temperatura mínima en el aite a 2 metros.
            """

            """
            Estos modelos tienen la capacidad de capturar patrones complejos en los datos y generar predicciones precisas. 
            Esto sugiere que los modelos LSTM pueden ser una herramienta útil para estudiar y predecir el clima en otras 
            regiones o para analizar diferentes variables climáticas.
            """

            """
            Los resultados obtenidos en este proyecto resaltan la importancia de monitorear y comprender los cambios en la 
            temperatura mínima en Colombia. Estos cambios pueden tener impactos significativos en diferentes aspectos, como 
            la agricultura, la salud pública y los ecosistemas. El análisis de la variación temporal de la temperatura 
            mínima proporciona información valiosa para comprender mejor el clima y tomar decisiones informadas 
            en áreas relacionadas.        
            """

            st.info(
                """
                [Código Fuente GitHub](https://github.com/fcastellanosp/UNIANDES_MINE_202310_AML/tree/main/Proyecto)
                """
            )
