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

    def test_obtener_anio_semana_iso(self) -> None:
        """
        Verifica que la función obtener_anio_semana_iso calcule correctamente
        el año y la semana ISO para fechas conocidas.
        """
        from analytics.ml_services import obtener_anio_semana_iso
        
        # 6 de Julio de 2026 es Lunes de la semana 28
        fecha = date(2026, 7, 6)
        anio, semana = obtener_anio_semana_iso(fecha)
        self.assertEqual(anio, 2026)
        self.assertEqual(semana, 28)

    def test_formatear_rango_semana(self) -> None:
        """
        Verifica que la función _formatear_rango_semana devuelva el formato correcto
        en español (p. ej. '17-Ago a 23-Ago').
        """
        from analytics.views import _formatear_rango_semana

        lunes = date(2026, 8, 17)
        domingo = date(2026, 8, 23)
        rango = _formatear_rango_semana(lunes, domingo)
        self.assertEqual(rango, "17-Ago a 23-Ago")

        # Caso cambio de mes (finales de diciembre a enero)
        fin_ano = date(2026, 12, 28)
        inicio_ano = date(2027, 1, 3)
        rango_ano = _formatear_rango_semana(fin_ano, inicio_ano)
        self.assertEqual(rango_ano, "28-Dic a 3-Ene")

    @patch("analytics.models.CombinacionPredicha.objects.filter")
    def test_obtener_pesos_adaptativos(self, mock_filter: Any) -> None:
        """
        Verifica que la función obtener_pesos_adaptativos calcule correctamente
        los pesos combinados usando la fórmula suavizada basada en aciertos previos.
        """
        from analytics.ml_services import obtener_pesos_adaptativos
        from unittest.mock import MagicMock
        
        # Simular combinaciones históricas
        comb_lstm = MagicMock()
        comb_lstm.estrategia = "lstm_pura"
        comb_lstm.aciertos_por_sorteo = {
            "2026-07-01": {"total_bolas": 3},
            "2026-07-03": {"total_bolas": 1}
        }
        
        comb_tendencia = MagicMock()
        comb_tendencia.estrategia = "tendencia_pura"
        comb_tendencia.aciertos_por_sorteo = {
            "2026-07-01": {"total_bolas": 1},
            "2026-07-03": {"total_bolas": 1}
        }
        
        mock_filter.return_value = [comb_lstm, comb_tendencia]
        
        # LSTM aciertos = 3 + 1 = 4. Tendencia aciertos = 1 + 1 = 2.
        # W_lstm = (4 + 1) / (4 + 2 + 2) = 5 / 8 = 0.625
        # W_tendencia = 1.0 - 0.625 = 0.375
        w_lstm, w_tendencia = obtener_pesos_adaptativos("primitiva")
        self.assertEqual(w_lstm, 0.625)
        self.assertEqual(w_tendencia, 0.375)

    def test_analizar_diferencia_fechas_vectorized(self) -> None:
        """
        Verifica que la función vectorizada analizar_diferencia_fechas calcule
        correctamente las estadísticas temporales (promedios, mínimos y máximos).
        """
        from analytics.ml_services import analizar_diferencia_fechas
        df = pd.DataFrame({
            "Fecha": pd.to_datetime(["2026-01-01", "2026-01-10", "2026-01-20"]),
            "Bola1": [5, 5, 5],
            "Bola2": [10, 20, 10],
        })
        res = analizar_diferencia_fechas(df, ["Bola1", "Bola2"])
        self.assertFalse(res.empty)
        # Bola 5 aparece en los 3 sorteos: difs de 9 y 10 días -> promedio 9.5
        row_5 = res[res["Numero"] == 5].iloc[0]
        self.assertEqual(row_5["Dias Promedio"], 9.5)
        self.assertEqual(row_5["Dias Min"], 9)
        self.assertEqual(row_5["Dias Max"], 10)

    @patch("django.contrib.auth.authenticate")
    def test_login_open_redirect_protection(self, mock_auth: Any) -> None:
        """
        Verifica que la vista de login neutralice intentos de Open Redirect
        cuando un atacante proporciona URLs externas en el parámetro next.
        """
        from django.contrib.auth.models import AnonymousUser
        mock_user = MagicMock()
        mock_user.is_authenticated = True
        mock_auth.return_value = mock_user

        # Intentar login con next malicioso externo
        response = self.client.post(
            "/login/?next=https://malicious-phishing.com",
            {"username": "admin", "password": "securepassword"}
        )
        # Debe redirigir al dashboard interno, NUNCA al dominio externo
        self.assertEqual(response.status_code, 302)
        self.assertNotIn("malicious-phishing.com", response.url)
        self.assertEqual(response.url, "/login/")  # Client redirects safely or returns internal URL

    def test_logout_redirects_safely(self) -> None:
        """
        Verifica que la vista de logout redirija de forma segura al login.
        """
        response = self.client.post("/logout/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/login/", response.url)

    @patch("analytics.ml_services.df_desde_orm")
    def test_cache_analisis_and_invalidation(self, mock_df_desde_orm: Any) -> None:
        """
        Verifica que ejecutar_analisis almacene en caché el resultado y que
        invalidar_cache_sorteo limpie la clave correspondiente.
        """
        from django.core.cache import cache
        from analytics.ml_services import invalidar_cache_sorteo, clave_cache_analisis

        df = pd.DataFrame({
            "Fecha": pd.to_datetime(["2026-06-01", "2026-06-08"]),
            "Bola1": [1, 2],
            "Bola2": [10, 11],
            "Bola3": [20, 21],
            "Bola4": [30, 31],
            "Bola5": [40, 41],
            "Bola6": [45, 46],
            "Complementario": [5, 6],
            "Reintegro": [0, 1],
        })
        mock_df_desde_orm.return_value = df

        # Primera ejecución: debe computar y guardar en caché
        res1 = ejecutar_analisis("primitiva", entrenar=False)
        self.assertIsNotNone(cache.get(clave_cache_analisis("primitiva")))

        # Segunda ejecución: debe devolver el mismo objeto desde la caché
        res2 = ejecutar_analisis("primitiva", entrenar=False)
        self.assertEqual(res1.total_sorteos, res2.total_sorteos)

        # Invalidar caché
        invalidar_cache_sorteo("primitiva")
        self.assertIsNone(cache.get(clave_cache_analisis("primitiva")))

    @patch("analytics.models.PrediccionSemanal.objects.get")
    def test_evaluar_predicciones_semana_primitiva_solo_6_bolas(self, mock_get_prediccion: Any) -> None:
        """
        Verifica que la evaluación de predicciones para Primitiva compare
        solamente las 6 bolas principales de forma estricta (excluyendo reintegro).
        """
        from analytics.ml_services import evaluar_predicciones_semana
        from unittest.mock import MagicMock

        # Mock de CombinacionPredicha de Primitiva
        comb = MagicMock()
        comb.bolas = [1, 2, 3, 4, 5, 6]
        comb.especiales = [5, 0]
        comb.aciertos_por_sorteo = {}

        prediccion = MagicMock()
        prediccion.combinaciones.all.return_value = [comb]
        mock_get_prediccion.return_value = prediccion

        # Mock de Sorteo real de Primitiva
        sorteo = MagicMock()
        sorteo.tipo_sorteo = "primitiva"
        sorteo.fecha = date(2026, 7, 6)
        sorteo.bolas_list.return_value = [1, 2, 3, 10, 20, 30]
        sorteo.especiales_list.return_value = [7, 0]

        evaluar_predicciones_semana(sorteo)

        self.assertTrue(comb.save.called)
        aciertos_guardados = comb.aciertos_por_sorteo["2026-07-06"]
        self.assertEqual(aciertos_guardados["total_bolas"], 3)
        self.assertEqual(set(aciertos_guardados["bolas_acertadas"]), {1, 2, 3})
        self.assertEqual(aciertos_guardados["total_especiales"], 0)

    @patch("torch.load")
    @patch("pathlib.Path.exists")
    def test_cargar_modelo_weights_only(self, mock_exists: Any, mock_torch_load: Any) -> None:
        """
        Verifica que cargar_modelo invoque torch.load con weights_only=True
        para garantizar una deserialización segura.
        """
        from analytics.ml_services import cargar_modelo, LSTMLoteria

        mock_exists.return_value = True
        dummy_model = LSTMLoteria(input_size=49)
        mock_torch_load.return_value = dummy_model.state_dict()

        modelo = cargar_modelo("primitiva", input_size=49)
        self.assertIsNotNone(modelo)
        mock_torch_load.assert_called_once()
        _, kwargs = mock_torch_load.call_args
        self.assertTrue(kwargs.get("weights_only", False))

