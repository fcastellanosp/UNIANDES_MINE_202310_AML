import Definitions
import numpy as np
import pandas as pd
import os.path as osp

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sodapy import Socrata


class DataController:

    def __init__(self, load_data=False):
        self.stations_ds_id = "n6vw-vkfe"
        self.temperature_ds_id = "sbwg-7ju4"
        self.open_data_host = "www.datos.gov.co"
        self.app_token = "qgHbkdSlEXEA5tsCEFeDngbpa"
        self.stations_df = None
        self.temperature_fields = ["codigoestacion", "fechaobservacion", "valorobservado"]
        self.prd_scaler = MinMaxScaler(feature_range=(0, 1))

        if load_data == True:
            self.query_all_stations()

    def get_y_coordinate(self, x):
        obj = x["ubicaci_n"]
        return obj["latitude"]

    def get_x_coordinate(self, x):
        obj = x["ubicaci_n"]
        return obj["longitude"]

    # Obtener el listado de estaciones
    def query_all_stations(self):
        client = Socrata(self.open_data_host, self.app_token)
        query = "categoria LIKE 'Climática%' AND estado = 'Activa'"
        results = client.get(self.stations_ds_id, where=query, limit=10000)

        # Convertir a pandas DataFrame
        self.stations_df = pd.DataFrame.from_records(results)
        self.stations_df['lon'] = self.stations_df.apply(self.get_x_coordinate, axis=1)
        self.stations_df['lat'] = self.stations_df.apply(self.get_y_coordinate, axis=1)
        self.stations_df["lon"] = self.stations_df["lon"].astype("float64")
        self.stations_df["lat"] = self.stations_df["lat"].astype("float64")
        self.stations_df["altitud"] = self.stations_df["altitud"].astype("float64")

        return self.stations_df

    # Obtener el listado de departamentos
    def query_dpto(self):
        if self.stations_df is None:
            self.query_all_stations()

        results = self.stations_df.groupby(['departamento'])['estado'].count().reset_index()
        results = results.sort_values(by='departamento', ascending=True)
        results = results.drop("estado", axis=1)
        return results

    # Obtener el listado de municipios dado un departamento
    def query_mun(self, dpto=""):
        if dpto == "":
            return

        if self.stations_df is None:
            self.query_all_stations()

        results = self.stations_df.copy()
        results = results[(results['departamento'] == dpto)]
        results = results.groupby(['municipio'])['estado'].count().reset_index()
        results = results.sort_values(by='municipio', ascending=True)
        results = results.drop("estado", axis=1)
        return results

    # Obtener el listado de municipios dado un departamento
    def query_stations_by_mun(self, mun=""):
        if mun == "":
            return

        if self.stations_df is None:
            self.query_all_stations()

        results = self.stations_df.copy()
        results = results[(results['municipio'] == mun)]
        results = results.groupby(['municipio'])['estado'].count().reset_index()
        results = results.sort_values(by='municipio', ascending=True)
        results = results.drop("estado", axis=1)
        return results

    def query_temp_station_values(self, station_code="0021205012", start_date="2020-01-01", ending_date="2020-04-30"):
        client = Socrata(self.open_data_host, self.app_token)
        query = f"codigoestacion='{station_code}' AND fechaobservacion BETWEEN '{start_date}' AND '{ending_date}'"
        results = client.get(self.temperature_ds_id, select=",".join(self.temperature_fields), where=query,
                             limit=200000)

        # Convertir a pandas DataFrame
        temp_station_df = pd.DataFrame.from_records(results)

        temp_station_df["fecha"] = pd.to_datetime(temp_station_df["fechaobservacion"]).dt.date
        temp_station_df["hora"] = pd.to_datetime(temp_station_df["fechaobservacion"]).dt.hour.astype('int32')
        # validation_station_df["hora"] = validation_station_df["hora"].astype('int32')
        temp_station_df["observacion"] = temp_station_df["valorobservado"].astype('float64')
        # temp_station_df["observacion_normalizada"] = self.prd_scaler.fit_transform(temp_station_df[["valorobservado"]])
        temp_station_df = temp_station_df.drop(['fechaobservacion', 'valorobservado'], axis=1)
        print(temp_station_df)
        return temp_station_df

    def predict(self, data, station_code="0021205012", hour=12):
        model_name = f"{station_code}_h{hour}"
        model_path = osp.join(Definitions.ROOT_DIR, "resources/models", f"{model_name}.h5")
        print(model_path)
        print(osp.exists(model_path))

        data["observacion_normalizada"] = self.prd_scaler.fit_transform(data[["observacion"]])

        data_v_df = pd.pivot_table(data, aggfunc='sum', columns='fecha',
                                                 index=['hora'], values='observacion_normalizada', fill_value=np.nan)

        data_v_df = data_v_df.fillna(method='ffill', axis=1)
        data_v_df = data_v_df.fillna(method='bfill', axis=1)

        input_dates = data_v_df.columns

        x_hour_real = data_v_df.loc[hour, input_dates].values.astype('float32')
        print(x_hour_real.shape)

        past, future = 8, 1
        x_real, x_real_lookback = self.create_dataset(x_hour_real, past)

        x_real_rdim = np.reshape(x_real, (x_real.shape[0], 1, x_real.shape[1]))
        print(x_real_rdim.shape)
        print(x_real_rdim)

        model_prd = load_model(model_path)
        # print(model_prd)
        # y_pred = model_prd.predict(X_real_rdim)
        # y_pred_ = self.prd_scaler.inverse_transform(y_pred)


        """ Función enargada de generar los dataset como línea de tiempo  """
    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)
