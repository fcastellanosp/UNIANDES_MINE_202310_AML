from sodapy import Socrata

import pandas as pd


class DataController:

    def __init__(self, load_data=False):
        self.stations_ds_id = "n6vw-vkfe"
        self.temperature_ds_id = "sbwg-7ju4"
        self.open_data_host = "www.datos.gov.co"
        self.app_token = "qgHbkdSlEXEA5tsCEFeDngbpa"
        self.stations_df = None

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
