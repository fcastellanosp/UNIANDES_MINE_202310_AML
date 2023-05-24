import Definitions
import math
import numpy as np
import pandas as pd
import os.path as osp

from keras.models import load_model
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
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
        #self.prd_scaler = MinMaxScaler(feature_range=(0, 1))

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
        print(f"Existe el modelo?: {osp.exists(model_path)}")

        print("Los datos")
        print(data)

        # data["observacion_normalizada"] = self.prd_scaler.fit_transform(data[["observacion"]])
        prd_scaler = MinMaxScaler(feature_range=(0, 1))
        prd_scaler.fit_transform(data[["observacion"]])
        print("Escalamiento OK!")

        # data["fecha"] = pd.to_datetime(data["fechaobservacion"])
        # data["hora"] = data["fecha"].dt.hour
        # data["hora"] = data["hora"].astype('int32')
        # data["observacion"] = data["observacion"].astype('float64')

        # data = data[data["hora"] == hour]
        print("Los datos antes del pivot")
        print(data)

        data_v_df = pd.pivot_table(data, aggfunc='sum', columns='fecha', index=['hora'],
                                   values='observacion', fill_value=np.nan)

        print("Los datos pivot")
        print(data_v_df)

        # Imputación
        data_v_df = data_v_df.fillna(method='ffill', axis=1)
        data_v_df = data_v_df.fillna(method='bfill', axis=1)

        input_dates = data_v_df.columns
        print("input_dates = ")
        print(input_dates)

        X_Real_val = data_v_df.loc[hour, input_dates].values.astype('float32')
        print("X_Real_val = ")
        print(X_Real_val)

        dataset = prd_scaler.fit_transform(np.reshape(X_Real_val, (-1, 1)))
        dataset = np.reshape(dataset, (-1))
        train, test = dataset[:-31], dataset[-46:]

        past, future = 8, 1
        X_train, Y_train = self.create_dataset(train, past)
        X_test, Y_test = self.create_dataset(test, past)
        print("X_train = ")
        print(X_train)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model_prd = load_model(model_path)

        # make predictions
        trainPredict = model_prd.predict(X_train)
        testPredict = model_prd.predict(X_test)
        metrics = pd.DataFrame(index=['Error Cuadrático Medio - MSE', 'Desviación media cuadrática - RMSE', 'Error absoluto medio - MAE', 'R2'], columns=['Entrenamiento', 'Prueba'])
        # invert predictions
        trainPredict = prd_scaler.inverse_transform(trainPredict)
        trainY = prd_scaler.inverse_transform([Y_train])
        testPredict = prd_scaler.inverse_transform(testPredict)
        testY = prd_scaler.inverse_transform([Y_test])
        # calculate mean squared error
        trainScore = mean_squared_error(trainY[0], trainPredict[:, 0])
        metrics.at['Error Cuadrático Medio - MSE', 'Entrenamiento'] = '{:.2f}'.format(trainScore)
        testScore = mean_squared_error(testY[0], testPredict[:, 0])
        metrics.at['Error Cuadrático Medio - MSE', 'Prueba'] = '{:.2f}'.format(testScore)
        # calculate root mean squared error
        trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
        metrics.at['Desviación media cuadrática - RMSE', 'Entrenamiento'] = '{:.2f}'.format(trainScore)
        testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
        metrics.at['Desviación media cuadrática - RMSE', 'Prueba'] = '{:.2f}'.format(testScore)
        # calculate r2
        trainScore = r2_score(trainY[0], trainPredict[:, 0])
        metrics.at['R2', 'Entrenamiento'] = '{:.2f}'.format(trainScore)
        testScore = r2_score(testY[0], testPredict[:, 0])
        metrics.at['R2', 'Prueba'] = '{:.2f}'.format(testScore)
        # calculate MAE
        trainScore = mean_absolute_error(trainY[0], trainPredict[:, 0])
        metrics.at['Error absoluto medio - MAE', 'Entrenamiento'] = '{:.2f}'.format(trainScore)
        testScore = mean_absolute_error(testY[0], testPredict[:, 0])
        metrics.at['Error absoluto medio - MAE', 'Prueba'] = '{:.2f}'.format(testScore)

        # shift train predictions for plotting
        trainPredictPlot = np.empty_like(dataset)
        trainPredictPlot[:] = np.nan
        trainPredictPlot[past:len(trainPredict) + past] = np.reshape(trainPredict, -1)
        # shift test predictions for plotting
        testPredictPlot = np.empty_like(dataset)
        testPredictPlot[:] = np.nan
        testPredictPlot[len(trainPredict):len(dataset) - 1] = np.reshape(testPredict, -1)

        title = f"Prediccción con {model_prd.name}, ventana [{past} días]"
        return title, input_dates, X_Real_val, trainPredictPlot, testPredictPlot, metrics

    """ Función enargada de generar los dataset como línea de tiempo  """
    def create_dataset(self, dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back):
            a = dataset[i:(i + look_back)]
            dataX.append(a)
            dataY.append(dataset[i + look_back])
        return np.array(dataX), np.array(dataY)
