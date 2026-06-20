from datetime import date
from unittest.mock import patch
from typing import Any

from django.test import TestCase
from django.test.runner import DiscoverRunner
import pandas as pd


class NoDbTestRunner(DiscoverRunner):
    """
    Ejecutor de pruebas personalizado que anula la creación de base de datos.
    Evita fallos de permisos en entornos PostgreSQL de solo lectura o sin privilegios de creación.
    """

    def setup_databases(self, **kwargs: Any) -> None:
        """
        Omite la configuración de base de datos.
        """
        pass

    def teardown_databases(self, old_config: Any, **kwargs: Any) -> None:
        """
        Omite la destrucción de base de datos.
        """
        pass


from analytics.ml_services import (
    _proxima_fecha,
    cargar_config_sorteos,
    config_por_tipo,
    ejecutar_analisis,
    tipos_disponibles,
)


class AnalyticsServicesTestCase(TestCase):
    """
    Suite de pruebas unitarias para validar las funciones de lógica analítica
    y predicción, así como redirecciones y renders en las vistas.
    """

    def test_cargar_config_sorteos(self) -> None:
        """
        Verifica que la configuración de los sorteos se cargue correctamente
        desde el archivo sorteos.json y devuelva una lista de diccionarios.
        """
        configs = cargar_config_sorteos()
        self.assertIsInstance(configs, list)
        self.assertGreater(len(configs), 0)

    def test_tipos_disponibles(self) -> None:
        """
        Comprueba que los tipos de sorteos principales estén definidos y disponibles.
        """
        tipos = tipos_disponibles()
        self.assertIn("primitiva", tipos)
        self.assertIn("gordo", tipos)
        self.assertIn("euromillones", tipos)

    def test_config_por_tipo(self) -> None:
        """
        Verifica que se obtenga la configuración correcta para un tipo de sorteo dado.
        """
        cfg = config_por_tipo("primitiva")
        self.assertEqual(cfg["sorteo"], "primitiva")
        self.assertIn("numeros", cfg)

    @patch("analytics.ml_services.df_desde_orm")
    def test_ejecutar_analisis_sin_entrenar(self, mock_df_desde_orm: Any) -> None:
        """
        Verifica que el flujo de ejecutar_analisis funcione correctamente sin
        requerir base de datos física simulando el DataFrame con datos ficticios.
        """
        # Crear un DataFrame ficticio para el test
        data = {
            "Fecha": pd.to_datetime(["2026-06-01", "2026-06-08", "2026-06-15"]),
            "Bola1": [1, 2, 1],
            "Bola2": [10, 11, 10],
            "Bola3": [20, 21, 20],
            "Bola4": [30, 31, 32],
            "Bola5": [40, 41, 42],
            "Bola6": [45, 46, 47],
            "Complementario": [5, 6, 7],
            "Reintegro": [0, 1, 2],
        }
        df = pd.DataFrame(data)
        for col in [
            "Bola1",
            "Bola2",
            "Bola3",
            "Bola4",
            "Bola5",
            "Bola6",
            "Complementario",
            "Reintegro",
        ]:
            df[col] = df[col].astype("Int64")

        mock_df_desde_orm.return_value = df

        # Ejecutar análisis sin entrenar (debe cargar el modelo existente o retornar vacío en lstm)
        resultado = ejecutar_analisis("primitiva", entrenar=False)
        self.assertEqual(resultado.total_sorteos, 3)
        self.assertFalse(resultado.frecuencias.empty)
        self.assertFalse(resultado.patrones_temporales.empty)
        self.assertFalse(resultado.indice_tendencia.empty)

    def test_proxima_fecha(self) -> None:
        """
        Verifica que la función _proxima_fecha devuelva una fecha válida posterior o igual a hoy.
        """
        prox = _proxima_fecha("primitiva")
        self.assertIsInstance(prox, date)
        self.assertGreaterEqual(prox, date.today())

    def test_dashboard_redirects_unauthenticated(self) -> None:
        """
        Comprueba que los usuarios no autenticados sean redirigidos a la pantalla de login.
        """
        response = self.client.get("/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    def test_login_page_renders(self) -> None:
        """
        Verifica que la página de login cargue de manera correcta con un código HTTP 200.
        """
        response = self.client.get("/login/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Iniciar Sesión")
