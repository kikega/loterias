from __future__ import annotations

import json
import logging
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
from django.core.cache import cache

from .models import Sorteo

logger = logging.getLogger(__name__)

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
    rows = list(qs.values_list("fecha", "bolas", "especiales"))
    if not rows:
        return pd.DataFrame()

    num_cols = cfg["numeros"]
    num_esp_cols = cfg["numeros_especiales"]
    len_num = len(num_cols)
    len_esp = len(num_esp_cols)

    registros: list[dict[str, Any]] = []
    for fecha, bolas, especiales in rows:
        bolas_list = bolas or []
        esp_list = especiales or []
        reg: dict[str, Any] = {"Fecha": pd.to_datetime(fecha)}
        for i in range(len_num):
            reg[num_cols[i]] = bolas_list[i] if i < len(bolas_list) else None
        for i in range(len_esp):
            reg[num_esp_cols[i]] = esp_list[i] if i < len(esp_list) else None
        registros.append(reg)

    df = pd.DataFrame.from_records(registros)
    for col in num_cols + num_esp_cols:
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
    Optimizado mediante vectorización completa con melt y groupby.

    Args:
        df (pd.DataFrame): Datos históricos del sorteo.
        columnas (list[str]): Columnas que representan los números sorteados.

    Returns:
        pd.DataFrame: Estadísticas temporales por número.
    """
    if df.empty:
        return pd.DataFrame()

    valid_cols = [c for c in columnas if c in df.columns]
    if not valid_cols:
        return pd.DataFrame()

    melted = df.melt(id_vars=["Fecha"], value_vars=valid_cols, value_name="Numero").dropna(subset=["Numero"])
    if melted.empty:
        return pd.DataFrame()

    melted["Numero"] = melted["Numero"].astype(int)
    melted = melted.drop_duplicates(subset=["Fecha", "Numero"]).sort_values("Fecha")
    melted["diff_days"] = melted.groupby("Numero")["Fecha"].diff().dt.days

    stats = (
        melted.groupby("Numero")
        .agg(
            Dias_Promedio=("diff_days", "mean"),
            Dias_Min=("diff_days", "min"),
            Dias_Max=("diff_days", "max"),
            Ultima_Aparicion=("Fecha", "max"),
            Conteo=("Fecha", "count"),
        )
        .reset_index()
    )

    stats = stats[stats["Conteo"] > 1].copy()
    if stats.empty:
        return pd.DataFrame()

    stats["Dias Promedio"] = stats["Dias_Promedio"].round(2)
    stats["Dias Min"] = stats["Dias_Min"].astype(int)
    stats["Dias Max"] = stats["Dias_Max"].astype(int)
    stats["Ultima Aparicion"] = stats["Ultima_Aparicion"]
    resultado = stats[["Numero", "Dias Promedio", "Dias Min", "Dias Max", "Ultima Aparicion"]].sort_values("Numero")
    resultado.reset_index(drop=True, inplace=True)
    return resultado


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
        epochs: int = 35,
        batch_size: int = 32,
        lr: float = 0.002,
        patience: int = 7,
    ) -> None:
        """
        Inicializa los hiperparámetros del optimizador y del ciclo de entrenamiento.

        Args:
            epochs (int, opcional): Número de épocas máximas.
            batch_size (int, opcional): Tamaño de lote.
            lr (float, opcional): Tasa de aprendizaje inicial.
            patience (int, opcional): Épocas de espera para parada temprana (early stopping).
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience


class LSTMLoteria(nn.Module):
    """
    Red Neuronal Recurrente bidireccional basada en LSTM con mecanismo de atención
    temporal y capas normalizadas para predecir probabilidades multietiqueta de números.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_capas: int = 2,
        dropout: float = 0.2,
    ) -> None:
        """
        Inicializa las capas LSTM bidireccionales, atención temporal y lineales.

        Args:
            input_size (int): Rango máximo de números posibles en el sorteo.
            hidden_size (int, opcional): Neuronas en las capas ocultas recurrentes.
            num_capas (int, opcional): Capas LSTM apiladas.
            dropout (float, opcional): Coeficiente de regularización dropout.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_capas,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_capas > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_size * 2, 1)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc_output = nn.Linear(hidden_size, input_size)

    def forward(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Ejecuta la propagación hacia adelante (forward pass) retornando logits no acotados.
        Diseñado para entrenamiento numéricamente estable con BCEWithLogitsLoss.

        Args:
            x_input (torch.Tensor): Tensores de entrada (Lote, Ventana, Números).

        Returns:
            torch.Tensor: Logits de salida para cada número.
        """
        lstm_out, _ = self.lstm(x_input)
        attn_weights = torch.softmax(self.attn(lstm_out), dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        hidden = self.act(self.ln(self.fc_hidden(context)))
        hidden = self.dropout(hidden)
        return self.fc_output(hidden)

    def predecir_probabilidades(self, x_input: torch.Tensor) -> torch.Tensor:
        """
        Calcula las probabilidades calibradas en el rango [0, 1] aplicando sigmoid sobre los logits.

        Args:
            x_input (torch.Tensor): Tensores de entrada.

        Returns:
            torch.Tensor: Probabilidades estimadas para cada número en [0, 1].
        """
        logits = self.forward(x_input)
        return torch.sigmoid(logits)


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
    val_split: float = 0.15,
) -> None:
    """
    Entrena los pesos del modelo LSTM utilizando pérdida ponderada por desbalance (BCEWithLogitsLoss),
    optimizador AdamW, corte de gradientes, scheduler y parada temprana (early stopping).

    Args:
        modelo (LSTMLoteria): Modelo a entrenar.
        x_tensor (torch.Tensor): Tensores de entrada.
        y_tensor (torch.Tensor): Tensores de salida objetivo.
        hp (HiperparametrosLSTM, opcional): Parámetros de épocas y optimizador.
        verbose (bool, opcional): Activa el reporte de pérdida por época en consola.
        val_split (float, opcional): Proporción de datos reservada para validación.
    """
    if hp is None:
        hp = HiperparametrosLSTM()

    torch.manual_seed(42)
    np.random.seed(42)

    n_samples = len(x_tensor)
    if n_samples < 20:
        return

    n_val = max(1, int(n_samples * val_split))
    n_train = n_samples - n_val

    x_train, y_train = x_tensor[:n_train], y_tensor[:n_train]
    x_val, y_val = x_tensor[n_train:], y_tensor[n_train:]

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=hp.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=hp.batch_size, shuffle=False)

    # Ponderación para clases positivas (corrige el sesgo de la baja probabilidad base)
    max_num = modelo.input_size
    cant_unos = y_tensor.sum(dim=1).mean().item()
    if cant_unos < 1.0:
        cant_unos = 6.0
    pos_weight_val = max(1.0, (max_num - cant_unos) / cant_unos)
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=hp.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-5
    )

    best_val_loss = float("inf")
    best_weights = None
    patience_counter = 0

    for epoch in range(1, hp.epochs + 1):
        modelo.train()
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = modelo(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)

        # Validación
        modelo.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for vx, vy in val_loader:
                v_logits = modelo(vx)
                v_loss = criterion(v_logits, vy)
                total_val_loss += v_loss.item()

        avg_val = total_val_loss / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_weights = {k: v.cpu().clone() for k, v in modelo.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == hp.epochs):
            print(
                f"  Época {epoch:2d}/{hp.epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} (mejor: {best_val_loss:.4f})"
            )

        if patience_counter >= hp.patience:
            if verbose:
                print(f"  Parada temprana (Early stopping) activada en época {epoch}.")
            break

    if best_weights is not None:
        modelo.load_state_dict(best_weights)
    modelo.eval()


def predecir_distribucion_completa_lstm(
    modelo: LSTMLoteria,
    ultima_secuencia: torch.Tensor,
) -> dict[int, float]:
    """
    Genera el mapa completo de probabilidades estimadas para todos los números posibles.

    Args:
        modelo (LSTMLoteria): Modelo de predicción entrenado.
        ultima_secuencia (torch.Tensor): Última secuencia observada (1, Ventana, Rango).

    Returns:
        dict[int, float]: Diccionario con mapeo de cada número a su probabilidad calibrada.
    """
    modelo.eval()
    with torch.no_grad():
        if hasattr(modelo, "predecir_probabilidades"):
            probs = modelo.predecir_probabilidades(ultima_secuencia).squeeze(0).cpu().numpy()
        else:
            out = modelo(ultima_secuencia).squeeze(0)
            probs = (torch.sigmoid(out) if out.max() > 1.0 or out.min() < 0.0 else out).cpu().numpy()
    return {i + 1: float(probs[i]) for i in range(modelo.input_size)}


def predecir_tendencias_lstm(
    modelo: LSTMLoteria,
    ultima_secuencia: torch.Tensor,
    top_k: int = 15,
) -> list[tuple[int, float]]:
    """
    Genera predicciones ordenadas de probabilidad de aparición para los top_k números.

    Args:
        modelo (LSTMLoteria): Modelo de predicción entrenado.
        ultima_secuencia (torch.Tensor): Última secuencia observada (1, Ventana, Rango).
        top_k (int, opcional): Número de predicciones principales a retornar.

    Returns:
        list[tuple[int, float]]: Pares (Número, Probabilidad estimada).
    """
    dist = predecir_distribucion_completa_lstm(modelo, ultima_secuencia)
    resultados = sorted(dist.items(), key=lambda par: par[1], reverse=True)
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
    Guarda el diccionario de estados del modelo en el disco de manera atómica
    escribiendo primero en un archivo temporal para evitar corrupción de pesos.

    Args:
        modelo (LSTMLoteria): Modelo entrenado a guardar.
        tipo_sorteo (str): Nombre identificativo del sorteo.
    """
    ruta = ruta_modelo_guardado(tipo_sorteo)
    temp_ruta = ruta.with_suffix(".tmp")
    torch.save(modelo.state_dict(), temp_ruta)
    temp_ruta.replace(ruta)


def cargar_modelo(tipo_sorteo: str, input_size: int) -> Optional[LSTMLoteria]:
    """
    Carga e inicializa el modelo de predicción LSTM guardado previamente en disco
    utilizando weights_only=True para prevenir riesgos de deserialización de código arbitrario.

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
        state_dict = torch.load(ruta, map_location="cpu", weights_only=True)
        modelo.load_state_dict(state_dict)
        modelo.eval()
        return modelo
    except Exception as e:
        logger.warning(f"No se pudieron cargar los pesos del modelo para {tipo_sorteo}: {e}")
        return None


# ---------------------------------------------------------------------------
# CACHÉ DE ANÁLISIS
# ---------------------------------------------------------------------------


def clave_cache_analisis(tipo_sorteo: str) -> str:
    """
    Genera la clave única de caché para los resultados del análisis de un sorteo.
    """
    return f"analisis_loteria_{tipo_sorteo}"


def invalidar_cache_sorteo(tipo_sorteo: str) -> None:
    """
    Invalida los datos cacheados de análisis y pesos adaptativos ante nuevos registros.
    """
    cache.delete(clave_cache_analisis(tipo_sorteo))
    logger.info(f"Caché invalidada para sorteo: {tipo_sorteo}")


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
        self.lstm_distribucion: dict[int, float] = {}
        self.frecuencias_especiales: pd.DataFrame = pd.DataFrame()


def ejecutar_analisis(
    tipo_sorteo: str, entrenar: bool = False
) -> ResultadoAnalisis:
    """
    Función orquestadora que ejecuta los análisis descriptivos de frecuencias, combinaciones,
    cadenas de Markov y predicciones neuronales de LSTM sobre el tipo de sorteo seleccionado.
    Utiliza una capa de caché en memoria de Django para respuestas instantáneas (<10ms).

    Args:
        tipo_sorteo (str): Nombre del sorteo (p. ej. 'primitiva', 'gordo').
        entrenar (bool, opcional): Permite forzar el reentrenamiento de la red LSTM (False por defecto).

    Returns:
        ResultadoAnalisis: Estructura con todos los datos e indicadores estadísticos.
    """
    clave = clave_cache_analisis(tipo_sorteo)
    if not entrenar:
        cached: Optional[ResultadoAnalisis] = cache.get(clave)
        if cached is not None:
            return cached

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
            resultado.lstm_distribucion = predecir_distribucion_completa_lstm(modelo, ultima_seq)
            resultado.lstm_top = predecir_tendencias_lstm(modelo, ultima_seq, top_k=15)

    if cols_esp:
        resultado.frecuencias_especiales = analizar_frecuencia_numeros(df, cols_esp)

    cache.set(clave, resultado, timeout=3600)
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

    # 1. Obtener puntuación completa de LSTM (todos los números, no sólo top 15)
    lstm_probs = getattr(resultado, "lstm_distribucion", {})
    if not lstm_probs and resultado.lstm_top:
        lstm_probs = {n: p for n, p in resultado.lstm_top}

    # 2. Obtener puntuación de Tendencia
    tendencia_scores = {}
    if not resultado.indice_tendencia.empty:
        for _, row in resultado.indice_tendencia.iterrows():
            tendencia_scores[int(row["Numero"])] = float(row["Indice"])

    # Normalizar puntuaciones para la estrategia híbrida
    max_t = max(tendencia_scores.values()) if tendencia_scores else 1.0
    min_t = min(tendencia_scores.values()) if tendencia_scores else 0.0
    rango_t = (max_t - min_t) if max_t != min_t else 1.0

    tendencia_norm = {
        num: (val - min_t) / rango_t for num, val in tendencia_scores.items()
    }

    # Obtener pesos adaptativos históricos
    w_lstm, w_tendencia = obtener_pesos_adaptativos(tipo_sorteo)

    # 3. Factor de coherencia Markoviana (paridad esperada según transición)
    prob_impar = 0.5
    markov_info = resultado.markov
    if "paridad" in markov_info and "distribucion" in markov_info["paridad"]:
        dist_p = markov_info["paridad"]["distribucion"]
        if hasattr(dist_p, "get"):
            prob_impar = float(dist_p.get("impar_dom", 0.5))

    todas_bolas_rango = list(range(limites["min_num"], limites["max_num"] + 1))

    # C. Estrategia Híbrida Adaptativa optimizada con Markov
    def score_hibrido(n: int) -> float:
        p_lstm = lstm_probs.get(n, 0.0)
        p_tend = tendencia_norm.get(n, 0.5)
        base = float(w_lstm * p_lstm + w_tendencia * p_tend)
        bono_m = ((prob_impar - 0.5) * 0.06) if (n % 2 != 0) else ((0.5 - prob_impar) * 0.06)
        return base + bono_m

    candidatos_hibridos = sorted(
        todas_bolas_rango,
        key=score_hibrido,
        reverse=True
    )

    # Las 3 apuestas de un mismo boleto cubren bolas distintas del pool más fuerte.
    # CORRECCIÓN ARQUITECTÓNICA: La combinación Híbrida Adaptativa (nuestro modelo estrella)
    # ahora elige EN PRIMER LUGAR sus mejores bolas, asegurando la máxima calidad predictiva.
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
    # 1. Híbrida Adaptativa elige primero del pool sus bolas favoritas
    bolas_hibridas = _elegir_desde_pool({n: score_hibrido(n) for n in restantes}, restantes)
    restantes = [n for n in restantes if n not in bolas_hibridas]
    # 2. LSTM elige sus favoritas disponibles
    bolas_lstm = _elegir_desde_pool(lstm_probs, restantes)
    restantes = [n for n in restantes if n not in bolas_lstm]
    # 3. Tendencia completa con sus preferidas del pool
    bolas_tendencia = _elegir_desde_pool(tendencia_scores, restantes)

    # 4. Generar números especiales combinando frecuencia histórica y retraso temporal (urgencia)
    especiales_sugeridos: list[list[int]] = [[], [], []]
    cfg = config_por_tipo(tipo_sorteo)
    cols_esp = cfg["numeros_especiales"]

    if cols_esp and not resultado.frecuencias_especiales.empty:
        freq_esp = resultado.frecuencias_especiales.copy()
        cant_esp = len(cols_esp)

        # Analizar retraso temporal / urgencia de los números especiales
        retrasos_esp: dict[int, pd.Timestamp] = {}
        df_esp = df_desde_orm(tipo_sorteo)
        if not df_esp.empty:
            for col_e in cols_esp:
                if col_e in df_esp.columns:
                    for _, r in df_esp[["Fecha", col_e]].dropna().iterrows():
                        val = int(r[col_e])
                        f_sorteo = r["Fecha"]
                        if val not in retrasos_esp or f_sorteo > retrasos_esp[val]:
                            retrasos_esp[val] = f_sorteo

        fecha_ref = pd.to_datetime(date.today())
        max_freq = freq_esp["Frecuencia"].max() if not freq_esp.empty else 1
        esp_scores = []
        for _, row in freq_esp.iterrows():
            num = int(row["Numero"])
            frec = int(row["Frecuencia"])
            ult = retrasos_esp.get(num, fecha_ref - pd.Timedelta(days=120))
            dias_sin_salir = max(0, (fecha_ref - ult).days)
            # Puntuación equilibrada: 55% retraso temporal (urgencia) + 45% frecuencia histórica
            score = (dias_sin_salir / 60.0) * 0.55 + (frec / max_freq) * 0.45
            esp_scores.append((num, score))

        esp_scores.sort(key=lambda x: x[1], reverse=True)
        esp_disponibles = [item[0] for item in esp_scores]

        esp_usados: set[int] = set()
        for i in range(3):
            seleccion = []
            for n in esp_disponibles:
                if n not in esp_usados:
                    esp_usados.add(n)
                    seleccion.append(n)
                    if len(seleccion) == cant_esp:
                        break
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
            logger.info(f"Iniciando reentrenamiento asíncrono para {tipo_sorteo}...")
            ejecutar_analisis(tipo_sorteo, entrenar=True)
            invalidar_cache_sorteo(tipo_sorteo)
            logger.info(f"Reentrenamiento completado con éxito para {tipo_sorteo}.")
        except Exception as e:
            logger.error(f"Error en reentrenamiento asíncrono para {tipo_sorteo}: {e}", exc_info=True)

    t = threading.Thread(target=_tarea_entrenamiento, daemon=True)
    t.start()

