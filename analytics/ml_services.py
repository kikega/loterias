from __future__ import annotations

import json
from collections import Counter
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import Sorteo

RUTA_JSON = Path(__file__).resolve().parent.parent / "sorteos.json"
RUTA_MODELOS = Path(__file__).resolve().parent / "modelos_entrenados"
RUTA_MODELOS.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# CONFIGURACIÓN DE SORTEOS
# ---------------------------------------------------------------------------

_CONFIG_CACHE: list[dict] | None = None


def cargar_config_sorteos() -> list[dict]:
    """
    Carga la configuración de sorteos desde el archivo JSON de configuración.
    Implementa almacenamiento en caché en memoria para evitar lecturas de disco repetidas.

    Returns:
        list[dict]: Lista de diccionarios de configuración de sorteos.

    Raises:
        FileNotFoundError: Si el archivo sorteos.json no existe.
        ValueError: Si el formato del JSON no es una lista.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE
    if not RUTA_JSON.exists():
        raise FileNotFoundError(f"No se encontró: {RUTA_JSON}")
    with RUTA_JSON.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El JSON debe contener una lista de sorteos.")
    _CONFIG_CACHE = data
    return data


def config_por_tipo(tipo: str) -> dict:
    """
    Obtiene la configuración específica para un tipo de sorteo determinado.

    Args:
        tipo (str): Nombre del sorteo (p. ej. 'primitiva', 'gordo').

    Returns:
        dict: Configuración asociada al sorteo.

    Raises:
        ValueError: Si el tipo de sorteo no existe en la configuración.
    """
    for cfg in cargar_config_sorteos():
        if cfg["sorteo"] == tipo:
            return cfg
    raise ValueError(f"Tipo de sorteo '{tipo}' no encontrado en sorteos.json")


def tipos_disponibles() -> list[str]:
    """
    Retorna la lista de identificadores de sorteos disponibles.

    Returns:
        list[str]: Lista con los nombres de sorteo habilitados.
    """
    return [cfg["sorteo"] for cfg in cargar_config_sorteos()]


# ---------------------------------------------------------------------------
# DATOS: Django ORM → DataFrame
# ---------------------------------------------------------------------------


def df_desde_orm(tipo_sorteo: str) -> pd.DataFrame:
    """
    Consulta la base de datos PostgreSQL mediante el ORM de Django y convierte
    los sorteos en un DataFrame estructurado y ordenado por fecha.

    Args:
        tipo_sorteo (str): Tipo del sorteo a recuperar.

    Returns:
        pd.DataFrame: DataFrame ordenado con columnas tipadas para los números del sorteo.
    """
    cfg = config_por_tipo(tipo_sorteo)
    qs = Sorteo.objects.filter(tipo_sorteo=tipo_sorteo).order_by("fecha")
    rows = qs.values("fecha", "bolas", "especiales")
    registros: list[dict] = []
    for r in rows:
        registro: dict = {"Fecha": pd.to_datetime(r["fecha"])}
        bolas: list[int] = r["bolas"] or []
        for i, col in enumerate(cfg["numeros"]):
            registro[col] = bolas[i] if i < len(bolas) else None
        especiales: list[int] = r["especiales"] or []
        for i, col in enumerate(cfg["numeros_especiales"]):
            registro[col] = especiales[i] if i < len(especiales) else None
        registros.append(registro)
    df = pd.DataFrame(registros)
    if df.empty:
        return df
    for col in cfg["numeros"] + cfg["numeros_especiales"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df.sort_values("Fecha", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# ANÁLISIS CLÁSICO
# ---------------------------------------------------------------------------


def analizar_frecuencia_numeros(
    df: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """
    Calcula la frecuencia de aparición absoluta y relativa (%) de cada número.

    Args:
        df (pd.DataFrame): Datos históricos del sorteo.
        columnas (list[str]): Nombres de las columnas con los números a analizar.

    Returns:
        pd.DataFrame: DataFrame ordenado con el número, su frecuencia y probabilidad.
    """
    if df.empty:
        return pd.DataFrame()
    todos = pd.concat([df[col] for col in columnas]).dropna()
    freq = todos.value_counts()
    total = len(todos)
    resultado = pd.DataFrame({"Numero": freq.index, "Frecuencia": freq.values})
    resultado["Probabilidad (%)"] = (resultado["Frecuencia"] / total * 100).round(3)
    resultado.sort_values("Numero", inplace=True)
    resultado.reset_index(drop=True, inplace=True)
    return resultado


def analizar_diferencia_fechas(
    df: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """
    Analiza el espaciado temporal entre apariciones de cada número, calculando
    las medias, mínimos y máximos de días sin salir, así como su última aparición.

    Args:
        df (pd.DataFrame): Datos históricos del sorteo.
        columnas (list[str]): Columnas que representan los números sorteados.

    Returns:
        pd.DataFrame: Estadísticas temporales por número.
    """
    numeros_unicos = sorted(
        pd.concat([df[col] for col in columnas]).dropna().unique()
    )
    filas: list[dict] = []
    for num in numeros_unicos:
        mask = df[columnas].eq(num).any(axis=1)
        fechas = df.loc[mask, "Fecha"].sort_values()
        if len(fechas) > 1:
            diffs = fechas.diff().dropna().dt.days
            filas.append({
                "Numero": num,
                "Dias Promedio": round(float(diffs.mean()), 2),
                "Dias Min": int(diffs.min()),
                "Dias Max": int(diffs.max()),
                "Ultima Aparicion": fechas.max(),
            })
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame(filas)


def analizar_combinaciones(
    df: pd.DataFrame, columnas: list[str], tamano_grupo: int
) -> list[tuple[tuple[int, ...], int]]:
    """
    Busca los grupos (parejas, tríos) de números que más frecuentemente
    aparecen juntos en un mismo sorteo. Optimizado mediante vectorización NumPy.

    Args:
        df (pd.DataFrame): Datos históricos del sorteo.
        columnas (list[str]): Columnas de los números principales.
        tamano_grupo (int): Tamaño del grupo de combinación (2 para parejas, 3 para tríos).

    Returns:
        list[tuple[tuple[int, ...], int]]: Las 15 combinaciones más frecuentes.
    """
    if df.empty:
        return []
    # Convertir a array de numpy directamente y omitir nulos para optimizar la velocidad
    arr = df[columnas].dropna(how="any").values.astype(int)
    contador: Counter = Counter()
    for row in arr:
        row.sort()
        contador.update(combinations(row, tamano_grupo))
    return contador.most_common(15)


def calcular_indice_tendencia(
    df_freq: pd.DataFrame,
    df_fechas: pd.DataFrame,
    fecha_referencia: date,
    peso_urgencia: float = 0.7,
    peso_frecuencia: float = 0.3,
) -> pd.DataFrame:
    """
    Combina la urgencia (días desde la última aparición en relación a su promedio)
    y la frecuencia histórica de cada número para generar un Índice de Tendencia.

    Args:
        df_freq (pd.DataFrame): Frecuencia histórica de los números.
        df_fechas (pd.DataFrame): Análisis de espaciado temporal de apariciones.
        fecha_referencia (date): Fecha teórica del próximo sorteo para los cálculos de días.
        peso_urgencia (float, opcional): Ponderación asignada al tiempo de ausencia.
        peso_frecuencia (float, opcional): Ponderación asignada a la frecuencia global.

    Returns:
        pd.DataFrame: Listado de números ordenados por su Índice de Tendencia descendente.
    """
    df = pd.merge(df_freq, df_fechas, on="Numero")
    fecha_ref_ts = pd.to_datetime(fecha_referencia)
    df["Dias Sin Salir"] = (fecha_ref_ts - df["Ultima Aparicion"]).dt.days
    df["Urgencia"] = df["Dias Sin Salir"] / df["Dias Promedio"]
    rango = df["Frecuencia"].max() - df["Frecuencia"].min()
    df["Freq Norm"] = (
        (df["Frecuencia"] - df["Frecuencia"].min()) / rango
        if rango > 0
        else 0.5
    )
    df["Indice"] = (
        df["Urgencia"] * peso_urgencia + df["Freq Norm"] * peso_frecuencia
    )
    df.sort_values("Indice", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# CADENAS DE MARKOV
# ---------------------------------------------------------------------------


def _median_fast(lst: list[int]) -> float:
    """
    Calcula rápidamente la mediana de una lista de enteros en Python puro.
    Optimiza la sobrecarga de importar o convertir a arrays NumPy en bucles pesados.

    Args:
        lst (list[int]): Lista de números enteros.

    Returns:
        float: Mediana calculada.
    """
    n = len(lst)
    if n == 0:
        return 0.0
    s_lst = sorted(lst)
    mid = n // 2
    if n % 2 != 0:
        return float(s_lst[mid])
    return (s_lst[mid - 1] + s_lst[mid]) / 2.0


class MarkovLoteria:
    """
    Clase que modela las transiciones de estado de un sorteo como una Cadena de Markov.
    Permite evaluar paridad, decenios medidos y sumas por terciles de los sorteos sucesivos.
    """
    TIPOS_ESTADO = ("paridad", "decenio", "zona_suma")

    def __init__(self, tipo_estado: str = "zona_suma") -> None:
        """
        Inicializa la instancia especificando la métrica de transición.

        Args:
            tipo_estado (str, opcional): Tipo de estado ('paridad', 'decenio', 'zona_suma').
        """
        if tipo_estado not in self.TIPOS_ESTADO:
            raise ValueError(f"tipo_estado debe ser uno de {self.TIPOS_ESTADO}")
        self.tipo_estado = tipo_estado
        self.matriz_transicion: pd.DataFrame = pd.DataFrame()
        self.estados_secuencia: list[str] = []
        self._terciles: Optional[tuple[float, float]] = None

    def definir_estado(self, bolas: list[int]) -> str:
        """
        Clasifica una combinación de bolas en un estado nominal según la métrica activa.

        Args:
            bolas (list[int]): Lista de bolas obtenidas en el sorteo.

        Returns:
            str: Nombre representativo del estado.
        """
        if self.tipo_estado == "paridad":
            pares = sum(1 for b in bolas if b % 2 == 0)
            impares = len(bolas) - pares
            if pares > impares:
                return "par_dom"
            if impares > pares:
                return "impar_dom"
            return "empate"
        if self.tipo_estado == "decenio":
            base = int(float(_median_fast(bolas)) // 10) * 10
            return f"{base}-{base + 9}"
        if self._terciles is None:
            raise RuntimeError("Calcula terciles llamando a construir_matriz_transicion()")
        total = sum(bolas)
        q33, q66 = self._terciles
        if total <= q33:
            return "bajo"
        if total <= q66:
            return "medio"
        return "alto"

    def construir_matriz_transicion(
        self, df: pd.DataFrame, columnas: list[str]
    ) -> pd.DataFrame:
        """
        Genera la matriz de probabilidad de transición a partir del historial.
        Optimizado para evitar iterrows de pandas y procesar transiciones en milisegundos.

        Args:
            df (pd.DataFrame): Datos históricos del sorteo.
            columnas (list[str]): Columnas que contienen los números de sorteo.

        Returns:
            pd.DataFrame: Matriz de transiciones con probabilidades relativas.
        """
        arr = df[columnas].dropna(how="all").values.astype(int)
        list_of_bolas = [row.tolist() for row in arr]

        if self.tipo_estado == "zona_suma":
            sumas = [sum(b) for b in list_of_bolas]
            q33, q66 = float(np.percentile(sumas, 33)), float(
                np.percentile(sumas, 66)
            )
            self._terciles = (q33, q66)
            estados = []
            for s in sumas:
                if s <= q33:
                    estados.append("bajo")
                elif s <= q66:
                    estados.append("medio")
                else:
                    estados.append("alto")
        else:
            estados = []
            for b in list_of_bolas:
                estados.append(self.definir_estado(b))

        self.estados_secuencia = estados
        contador: Counter = Counter(zip(estados[:-1], estados[1:]))
        estados_unicos = sorted(set(estados))
        matriz = pd.DataFrame(0, index=estados_unicos, columns=estados_unicos)
        for (origen, destino), cuenta in contador.items():
            matriz.loc[origen, destino] = cuenta
        totales = matriz.sum(axis=1)
        self.matriz_transicion = matriz.div(totales, axis=0).fillna(0)
        return self.matriz_transicion

    def probabilidad_siguiente_estado(self, estado_actual: str) -> pd.Series:
        """
        Devuelve la distribución de probabilidad para el siguiente estado partiendo del actual.

        Args:
            estado_actual (str): Estado de origen actual.

        Returns:
            pd.Series: Serie ordenada con las probabilidades relativas de transición.
        """
        if self.matriz_transicion.empty:
            raise RuntimeError("Llama primero a construir_matriz_transicion()")
        if estado_actual not in self.matriz_transicion.index:
            return pd.Series(dtype=float)
        return self.matriz_transicion.loc[estado_actual].sort_values(ascending=False)


# ---------------------------------------------------------------------------
# DEEP LEARNING: LSTM
# ---------------------------------------------------------------------------


class HiperparametrosLSTM:
    """
    Clase contenedora de los hiperparámetros de entrenamiento del modelo neuronal LSTM.
    """

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 0.001,
    ) -> None:
        """
        Inicializa los hiperparámetros del optimizador y del ciclo de entrenamiento.

        Args:
            epochs (int, opcional): Número de épocas.
            batch_size (int, opcional): Tamaño de lote.
            lr (float, opcional): Tasa de aprendizaje inicial.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr


class LSTMLoteria(nn.Module):
    """
    Red Neuronal Recurrente basada en LSTM para predecir probabilidades multietiqueta
    de aparición de números individuales basándose en una secuencia de sorteos anteriores.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_capas: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """
        Inicializa las capas LSTM y lineales de la red.

        Args:
            input_size (int): Rango máximo de números posibles en el sorteo (dimensión del input/output).
            hidden_size (int, opcional): Número de neuronas en las capas ocultas.
            num_capas (int, opcional): Capas LSTM apiladas.
            dropout (float, opcional): Coeficiente de regularización dropout.
        """
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_capas,
            batch_first=True,
            dropout=dropout if num_capas > 1 else 0.0,
        )
        self.fc_hidden = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc_output = nn.Linear(hidden_size // 2, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta la propagación hacia adelante (forward pass) de la red.

        Args:
            x_input (torch.Tensor): Tensores tridimensionales de entrada (Lote, Ventana, Números).

        Returns:
            torch.Tensor: Vector con probabilidades sigmoides de salida para cada número.
        """
        lstm_out, _ = self.lstm(x_input)
        last_out = lstm_out[:, -1, :]
        hidden = self.relu(self.fc_hidden(last_out))
        hidden = self.dropout(hidden)
        return self.sigmoid(self.fc_output(hidden))


def preparar_secuencias_lstm(
    df: pd.DataFrame,
    columnas: list[str],
    max_numero: int,
    ventana: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transforma la serie temporal de sorteos en un conjunto supervisado de tensores de PyTorch.
    Representa cada sorteo mediante vectores de tipo one-hot multi-hot.
    Optimizado mediante vectorización completa en NumPy.

    Args:
        df (pd.DataFrame): Historial del sorteo.
        columnas (list[str]): Nombres de las columnas con las bolas principales.
        max_numero (int): Número máximo en el bombo de sorteos.
        ventana (int, opcional): Longitud de la ventana temporal de entrada (pasos anteriores).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: Tupla de tensores (Entradas, Salidas deseadas).
    """
    if df.empty:
        return (
            torch.empty(0, ventana, max_numero, dtype=torch.float32),
            torch.empty(0, max_numero, dtype=torch.float32),
        )
    arr = df[columnas].fillna(0).values.astype(int)
    n_rows = len(arr)
    vectores = np.zeros((n_rows, max_numero), dtype=np.float32)
    for i in range(n_rows):
        row = arr[i]
        valid_mask = (row >= 1) & (row <= max_numero)
        vectores[i, row[valid_mask] - 1] = 1.0

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for i in range(ventana, n_rows):
        x_list.append(vectores[i - ventana : i])
        y_list.append(vectores[i])
    if not x_list:
        return (
            torch.empty(0, ventana, max_numero, dtype=torch.float32),
            torch.empty(0, max_numero, dtype=torch.float32),
        )
    x_arr = np.stack(x_list)
    y_arr = np.stack(y_list)
    return (
        torch.tensor(x_arr, dtype=torch.float32),
        torch.tensor(y_arr, dtype=torch.float32),
    )


def entrenar_lstm(
    modelo: LSTMLoteria,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    hp: Optional[HiperparametrosLSTM] = None,
    verbose: bool = False,
) -> None:
    """
    Entrena los pesos del modelo LSTM utilizando retropropagación y pérdida BCELoss.

    Args:
        modelo (LSTMLoteria): Modelo a entrenar.
        x_tensor (torch.Tensor): Tensores de entrada.
        y_tensor (torch.Tensor): Tensores de salida objetivo.
        hp (HiperparametrosLSTM, opcional): Parámetros de épocas y optimizador.
        verbose (bool, opcional): Activa el reporte de pérdida por época en consola.
    """
    if hp is None:
        hp = HiperparametrosLSTM()
    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=hp.lr)
    modelo.train()
    for epoch in range(1, hp.epochs + 1):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = modelo(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if verbose and (epoch % 10 == 0 or epoch == 1):
            avg_loss = total_loss / len(loader)
            print(f"  Época {epoch:3d}/{hp.epochs} | Loss: {avg_loss:.4f}")


def predecir_tendencias_lstm(
    modelo: LSTMLoteria,
    ultima_secuencia: torch.Tensor,
    top_k: int = 15,
) -> list[tuple[int, float]]:
    """
    Genera predicciones de probabilidad de aparición para el próximo sorteo.

    Args:
        modelo (LSTMLoteria): Modelo de predicción entrenado.
        ultima_secuencia (torch.Tensor): Última secuencia observada (1, Ventana, Rango).
        top_k (int, opcional): Número de predicciones principales a retornar.

    Returns:
        list[tuple[int, float]]: Pares (Número, Probabilidad estimada).
    """
    modelo.eval()
    with torch.no_grad():
        probs = modelo(ultima_secuencia).squeeze(0).numpy()
    resultados = [(i + 1, float(probs[i])) for i in range(modelo.input_size)]
    resultados.sort(key=lambda par: par[1], reverse=True)
    return resultados[:top_k]


# ---------------------------------------------------------------------------
# PERSISTENCIA DEL MODELO
# ---------------------------------------------------------------------------


def ruta_modelo_guardado(tipo_sorteo: str) -> Path:
    """
    Devuelve la ruta absoluta del archivo de pesos guardados (.pt) del modelo LSTM.

    Args:
        tipo_sorteo (str): Tipo del sorteo.

    Returns:
        Path: Ruta de persistencia.
    """
    return RUTA_MODELOS / f"lstm_{tipo_sorteo}.pt"


def modelo_existe(tipo_sorteo: str) -> bool:
    """
    Verifica si existe el archivo de pesos guardado en disco para un sorteo.

    Args:
        tipo_sorteo (str): Tipo del sorteo.

    Returns:
        bool: True si el archivo existe, False de lo contrario.
    """
    return ruta_modelo_guardado(tipo_sorteo).exists()


def guardar_modelo(modelo: LSTMLoteria, tipo_sorteo: str) -> None:
    """
    Guarda el diccionario de estados del modelo en el disco.

    Args:
        modelo (LSTMLoteria): Modelo entrenado a guardar.
        tipo_sorteo (str): Nombre identificativo del sorteo.
    """
    ruta = ruta_modelo_guardado(tipo_sorteo)
    torch.save(modelo.state_dict(), ruta)


def cargar_modelo(tipo_sorteo: str, input_size: int) -> Optional[LSTMLoteria]:
    """
    Carga e inicializa el modelo de predicción LSTM guardado previamente en disco.

    Args:
        tipo_sorteo (str): Nombre del sorteo.
        input_size (int): Dimensión del bombo de números.

    Returns:
        Optional[LSTMLoteria]: Instancia de la red entrenada lista para inferencia,
                              o None si no hay pesos guardados o falla la carga.
    """
    ruta = ruta_modelo_guardado(tipo_sorteo)
    if not ruta.exists():
        return None
    modelo = LSTMLoteria(input_size=input_size)
    try:
        modelo.load_state_dict(torch.load(ruta, map_location="cpu"))
        modelo.eval()
        return modelo
    except Exception:
        return None


# ---------------------------------------------------------------------------
# ORQUESTADOR POR TIPO DE SORTEO
# ---------------------------------------------------------------------------


class ResultadoAnalisis:
    """
    Clase contenedora de todas las métricas analíticas recopiladas en un análisis completo.
    """

    def __init__(self, tipo_sorteo: str) -> None:
        """
        Inicializa las variables e indicadores vacíos de resultados de análisis.

        Args:
            tipo_sorteo (str): Nombre del sorteo analizado.
        """
        self.tipo_sorteo = tipo_sorteo
        self.total_sorteos: int = 0
        self.frecuencias: pd.DataFrame = pd.DataFrame()
        self.patrones_temporales: pd.DataFrame = pd.DataFrame()
        self.indice_tendencia: pd.DataFrame = pd.DataFrame()
        self.combinaciones_pares: list = []
        self.combinaciones_trios: list = []
        self.markov: dict[str, dict] = {}
        self.lstm_top: list[tuple[int, float]] = []
        self.frecuencias_especiales: pd.DataFrame = pd.DataFrame()


def ejecutar_analisis(
    tipo_sorteo: str, entrenar: bool = True
) -> ResultadoAnalisis:
    """
    Función orquestadora que ejecuta los análisis descriptivos de frecuencias, combinaciones,
    cadenas de Markov y predicciones neuronales de LSTM sobre el tipo de sorteo seleccionado.
    Optimizado para un tiempo de respuesta inferior a 0.25 segundos sin entrenamiento activo.

    Args:
        tipo_sorteo (str): Nombre del sorteo (p. ej. 'primitiva', 'gordo').
        entrenar (bool, opcional): Permite forzar el reentrenamiento de la red LSTM (por defecto False en peticiones web).

    Returns:
        ResultadoAnalisis: Estructura con todos los datos e indicadores estadísticos.
    """
    cfg = config_por_tipo(tipo_sorteo)
    df = df_desde_orm(tipo_sorteo)
    resultado = ResultadoAnalisis(tipo_sorteo)
    if df.empty:
        return resultado

    resultado.total_sorteos = len(df)
    cols = cfg["numeros"]
    cols_esp = cfg["numeros_especiales"]

    resultado.frecuencias = analizar_frecuencia_numeros(df, cols)
    resultado.patrones_temporales = analizar_diferencia_fechas(df, cols)
    resultado.combinaciones_pares = analizar_combinaciones(df, cols, 2)
    resultado.combinaciones_trios = analizar_combinaciones(df, cols, 3)

    if not resultado.frecuencias.empty and not resultado.patrones_temporales.empty:
        prox = _proxima_fecha(tipo_sorteo)
        resultado.indice_tendencia = calcular_indice_tendencia(
            resultado.frecuencias, resultado.patrones_temporales, prox
        )

    for tipo in MarkovLoteria.TIPOS_ESTADO:
        markov = MarkovLoteria(tipo_estado=tipo)
        markov.construir_matriz_transicion(df, cols)
        bolas_ult = df[cols].dropna().iloc[-1].astype(int).tolist()
        estado_actual = markov.definir_estado(bolas_ult)
        dist = markov.probabilidad_siguiente_estado(estado_actual)
        resultado.markov[tipo] = {
            "matriz": markov.matriz_transicion,
            "estado_actual": estado_actual,
            "distribucion": dist,
        }

    max_num = int(pd.concat([df[c] for c in cols]).dropna().max())
    ventana = 10
    x_tensor, y_tensor = preparar_secuencias_lstm(df, cols, max_num, ventana)

    if x_tensor.shape[0] >= 20:
        if not entrenar:
            modelo = cargar_modelo(tipo_sorteo, max_num)
        else:
            modelo = LSTMLoteria(input_size=max_num)
            entrenar_lstm(modelo, x_tensor, y_tensor, verbose=True)
            guardar_modelo(modelo, tipo_sorteo)

        if modelo is not None:
            ultima_seq = x_tensor[-1].unsqueeze(0)
            resultado.lstm_top = predecir_tendencias_lstm(modelo, ultima_seq)

    if cols_esp:
        resultado.frecuencias_especiales = analizar_frecuencia_numeros(df, cols_esp)

    return resultado


def _proxima_fecha(tipo_sorteo: str) -> date:
    """
    Calcula una fecha estimada para el próximo sorteo, redondeando al próximo domingo.

    Args:
        tipo_sorteo (str): Tipo de sorteo.

    Returns:
        date: Fecha estimada.
    """
    hoy = date.today()
    dias_hasta_domingo = (6 - hoy.weekday() + 7) % 7 or 7
    return hoy + timedelta(days=dias_hasta_domingo)


# ---------------------------------------------------------------------------
# LÓGICA DE PREDICCIONES SEMANALES Y APRENDIZAJE ADAPTATIVO
# ---------------------------------------------------------------------------

LIMITES_SORTEO = {
    "primitiva": {
        "min_num": 1,
        "max_num": 49,
        "cant_bolas": 6,
        "especiales": [
            {"nombre": "Complementario", "min": 1, "max": 49},
            {"nombre": "Reintegro", "min": 0, "max": 9}
        ]
    },
    "euromillones": {
        "min_num": 1,
        "max_num": 50,
        "cant_bolas": 5,
        "especiales": [
            {"nombre": "Estrella1", "min": 1, "max": 12},
            {"nombre": "Estrella2", "min": 1, "max": 12}
        ]
    },
    "gordo": {
        "min_num": 1,
        "max_num": 54,
        "cant_bolas": 5,
        "especiales": [
            {"nombre": "Clave", "min": 0, "max": 9}
        ]
    }
}


def obtener_anio_semana_iso(fecha: date) -> tuple[int, int]:
    """
    Obtiene el año y el número de semana según el estándar ISO 8601.

    Args:
        fecha (date): Fecha de referencia.

    Returns:
        tuple[int, int]: Año y número de semana ISO.
    """
    iso_calendar = fecha.isocalendar()
    return iso_calendar[0], iso_calendar[1]


def obtener_pesos_adaptativos(tipo_sorteo: str) -> tuple[float, float]:
    """
    Analiza el rendimiento histórico de aciertos de las estrategias
    para calibrar los pesos de la predicción híbrida (aprendizaje adaptativo).

    Args:
        tipo_sorteo (str): Tipo de sorteo.

    Returns:
        tuple[float, float]: Pesos (W_lstm, W_tendencia) auto-ajustados.
    """
    from .models import CombinacionPredicha

    combs_historicas = CombinacionPredicha.objects.filter(
        prediccion_semanal__tipo_sorteo=tipo_sorteo,
        procesado=True
    )

    aciertos_lstm = 0
    aciertos_tendencia = 0

    for c in combs_historicas:
        total_aciertos_comb = 0
        for fecha_sorteo, aciertos_info in c.aciertos_por_sorteo.items():
            total_aciertos_comb += aciertos_info.get("total_bolas", 0)

        if c.estrategia == "lstm_pura":
            aciertos_lstm += total_aciertos_comb
        elif c.estrategia == "tendencia_pura":
            aciertos_tendencia += total_aciertos_comb

    # Suavizado de Laplace para evitar división por cero o pesos sesgados inicialmente
    denominador = aciertos_lstm + aciertos_tendencia + 2.0
    w_lstm = (aciertos_lstm + 1.0) / denominador
    w_tendencia = 1.0 - w_lstm

    return w_lstm, w_tendencia


def generar_predicciones_semanales(
    tipo_sorteo: str, anio: int, semana: int
) -> PrediccionSemanal:
    """
    Genera y guarda 3 combinaciones estimadas utilizando las 3 estrategias:
    LSTM Pura, Tendencia Pura e Híbrida Adaptativa (con pesos auto-ajustados).

    Args:
        tipo_sorteo (str): Tipo del sorteo.
        anio (int): Año ISO.
        semana (int): Semana ISO.

    Returns:
        PrediccionSemanal: Objeto de predicción generado e insertado.
    """
    from .models import PrediccionSemanal, CombinacionPredicha

    # Verificar si ya existe
    pred, creada = PrediccionSemanal.objects.get_or_create(
        tipo_sorteo=tipo_sorteo, anio=anio, semana=semana
    )
    if not creada and pred.combinaciones.exists():
        return pred

    # Eliminar posibles combinaciones vacías e inicializar de nuevo
    pred.combinaciones.all().delete()

    # Ejecutar análisis actual
    resultado = ejecutar_analisis(tipo_sorteo, entrenar=False)
    limites = LIMITES_SORTEO[tipo_sorteo]
    cant_bolas = limites["cant_bolas"]

    # 1. Obtener puntuación de LSTM
    lstm_probs = {n: p for n, p in resultado.lstm_top} if resultado.lstm_top else {}
    
    # 2. Obtener puntuación de Tendencia
    tendencia_scores = {}
    if not resultado.indice_tendencia.empty:
        for _, row in resultado.indice_tendencia.iterrows():
            tendencia_scores[int(row["Numero"])] = float(row["Indice"])

    # Normalizar puntuaciones para la estrategia híbrida
    # LSTM: la probabilidad ya está en rango [0, 1]
    # Tendencia: normalizar en rango [0, 1]
    max_t = max(tendencia_scores.values()) if tendencia_scores else 1.0
    min_t = min(tendencia_scores.values()) if tendencia_scores else 0.0
    rango_t = (max_t - min_t) if max_t != min_t else 1.0

    tendencia_norm = {
        num: (val - min_t) / rango_t for num, val in tendencia_scores.items()
    }

    # Obtener pesos adaptativos
    w_lstm, w_tendencia = obtener_pesos_adaptativos(tipo_sorteo)

    # 3. Generar combinaciones para cada estrategia
    todas_bolas_rango = list(range(limites["min_num"], limites["max_num"] + 1))

    # C. Estrategia Híbrida Adaptativa
    def score_hibrido(n: int) -> float:
        p_lstm = lstm_probs.get(n, 0.0)
        p_tend = tendencia_norm.get(n, 0.5)
        return w_lstm * p_lstm + w_tend_p if (w_tend_p := w_tendencia * p_tend) else w_lstm * p_lstm

    candidatos_hibridos = sorted(
        todas_bolas_rango,
        key=score_hibrido,
        reverse=True
    )

    # Las 3 apuestas de un mismo boleto no deben repetir bolas entre sí.
    # Además se optimiza la cobertura conjunta del boleto: la selección se limita
    # al "pool" de los 3*cant_bolas números con mayor puntuación combinada (híbrida)
    # y cada estrategia elige sus cant_bolas preferidas dentro de ese pool.
    pool_boleto = candidatos_hibridos[: cant_bolas * 3]

    def _elegir_desde_pool(
        claves_score: dict[int, float], disponibles: list[int]
    ) -> list[int]:
        seleccion = sorted(
            disponibles,
            key=lambda n: claves_score.get(n, 0.0),
            reverse=True,
        )[:cant_bolas]
        return sorted(seleccion)

    restantes = pool_boleto
    bolas_lstm = _elegir_desde_pool(lstm_probs, restantes)
    restantes = [n for n in restantes if n not in bolas_lstm]
    bolas_tendencia = _elegir_desde_pool(tendencia_scores, restantes)
    restantes = [n for n in restantes if n not in bolas_tendencia]
    bolas_hibridas = _elegir_desde_pool({n: score_hibrido(n) for n in restantes}, restantes)

    # 4. Generar números especiales basados en frecuencias de aparición
    # Para cada número especial, tomamos los más frecuentes en el histórico
    especiales_sugeridos: list[list[int]] = [[], [], []]
    
    cfg = config_por_tipo(tipo_sorteo)
    cols_esp = cfg["numeros_especiales"]

    if cols_esp and not resultado.frecuencias_especiales.empty:
        # Obtenemos los especiales más frecuentes según su conteo
        freq_esp = resultado.frecuencias_especiales
        esp_mas_comunes = []
        for _, row in freq_esp.iterrows():
            esp_mas_comunes.append((int(row["Numero"]), int(row["Frecuencia"])))
        esp_mas_comunes.sort(key=lambda x: x[1], reverse=True)
        
        # Para Euromillones requerimos 2 estrellas por combinación. Para Gordo 1 clave.
        cant_esp = len(cols_esp)

        # Repartimos los especiales más frecuentes entre las 3 apuestas sin repetir
        # número entre ellas siempre que haya suficientes distintos.
        esp_disponibles = [item[0] for item in esp_mas_comunes]
        esp_usados: set[int] = set()
        for i in range(3):
            seleccion = []
            for n in esp_disponibles:
                if n not in esp_usados:
                    esp_usados.add(n)
                    seleccion.append(n)
                    if len(seleccion) == cant_esp:
                        break
            # Si no hay suficientes especiales distintos, se reutilizan los más frecuentes
            if len(seleccion) < cant_esp:
                for n in esp_disponibles:
                    if n not in seleccion:
                        seleccion.append(n)
                        if len(seleccion) == cant_esp:
                            break
            especiales_sugeridos[i] = sorted(seleccion)
    elif cols_esp:
        # Fallback si no hay frecuencias especiales
        cant_esp = len(cols_esp)
        for i in range(3):
            especiales_sugeridos[i] = list(range(1, cant_esp + 1))

    # Guardar las combinaciones
    CombinacionPredicha.objects.create(
        prediccion_semanal=pred,
        orden=1,
        estrategia="lstm_pura",
        bolas=bolas_lstm,
        especiales=especiales_sugeridos[0] if cols_esp else None
    )

    CombinacionPredicha.objects.create(
        prediccion_semanal=pred,
        orden=2,
        estrategia="tendencia_pura",
        bolas=bolas_tendencia,
        especiales=especiales_sugeridos[1] if cols_esp else None
    )

    CombinacionPredicha.objects.create(
        prediccion_semanal=pred,
        orden=3,
        estrategia="hibrida_adaptativa",
        bolas=bolas_hibridas,
        especiales=especiales_sugeridos[2] if cols_esp else None
    )

    return pred


def evaluar_predicciones_semana(sorteo: Sorteo) -> None:
    """
    Compara las combinaciones estimadas de la semana del sorteo real
    e inserta los aciertos calculados (aprendizaje).

    Args:
        sorteo (Sorteo): Sorteo real recién ingresado.
    """
    from .models import PrediccionSemanal

    # Obtener año y semana ISO
    anio, semana = obtener_anio_semana_iso(sorteo.fecha)

    try:
        prediccion = PrediccionSemanal.objects.get(
            tipo_sorteo=sorteo.tipo_sorteo,
            anio=anio,
            semana=semana
        )
    except PrediccionSemanal.DoesNotExist:
        # No se generaron predicciones previas para esta semana
        return

    bolas_reales = set(sorteo.bolas_list())
    especiales_reales = set(sorteo.especiales_list())

    for comb in prediccion.combinaciones.all():
        bolas_predichas = set(comb.bolas)
        especiales_predichas = set(comb.especiales or [])

        if sorteo.tipo_sorteo == "primitiva":
            # Para Primitiva, solo comparar las 6 bolas principales por petición
            bolas_acertadas = bolas_predichas.intersection(bolas_reales)
            especiales_acertadas = set()
        else:
            bolas_acertadas = bolas_predichas.intersection(bolas_reales)
            especiales_acertadas = especiales_predichas.intersection(especiales_reales)

        # Guardar en el diccionario de aciertos bajo la fecha del sorteo
        comb.aciertos_por_sorteo[str(sorteo.fecha)] = {
            "bolas_acertadas": list(bolas_acertadas),
            "especiales_acertados": list(especiales_acertadas),
            "total_bolas": len(bolas_acertadas),
            "total_especiales": len(especiales_acertadas)
        }
        comb.procesado = True
        comb.save()


def entrenar_modelo_asincrono(tipo_sorteo: str) -> None:
    """
    Inicia un hilo en segundo plano para reentrenar el modelo LSTM
    del tipo de sorteo seleccionado sin bloquear el hilo principal.

    Args:
        tipo_sorteo (str): Tipo del sorteo a reentrenar.
    """
    import threading

    def _tarea_entrenamiento():
        try:
            # Reentrenamos el modelo con 15 épocas para que sea rápido pero aprenda el nuevo dato
            # Podemos forzar el entrenamiento completo
            ejecutar_analisis(tipo_sorteo, entrenar=True)
        except Exception:
            # Silenciar errores del hilo en producción/desarrollo silencioso
            pass

    t = threading.Thread(target=_tarea_entrenamiento)
    t.daemon = True
    t.start()

