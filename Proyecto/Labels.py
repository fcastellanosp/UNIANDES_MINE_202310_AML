from datetime import datetime

class Labels:

    def __init__(self):
        """ Constructor """
        self.main_title = "Pronóstico de temperatura del aire a 2 metros"
        self.main_description = """Este es el pronóstico de la temperatura del aire basado en los datos del IDEAM. """
        self.form_title = "Parametros"
        self.form_subtitle = f"Para lanzar la app, por favor seleccione la siguiente información"
        self.processing = "Procesando datos...."
        self.from_init_date = "Fecha Inicial"
        self.form_end_date = "Fecha Final"
        self.subtitle = "A continuación los resultados..."
        self.default_state = "State"

        date_format = "%Y/%m/%d"
        self.default_init_date = datetime.strptime("2023/01/01", date_format)
        self.default_ending_date = datetime.strptime(datetime.now().strftime(date_format), date_format)
