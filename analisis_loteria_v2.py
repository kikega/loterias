"""
analisis_loteria_v2.py
======================
Script educativo para el análisis estadístico de sorteos de lotería.

AVISO: Este script calcula TENDENCIAS ESTADÍSTICAS basadas en históricos.
Los sorteos son eventos independientes. Ningún modelo matemático puede
predecir el resultado de un sorteo real.

Dependencias:
    pip install pandas numpy torch pydantic pydantic-settings \
                sqlalchemy psycopg2-binary python-dotenv
"""

from __future__ import annotations

import warnings
from collections import Counter
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# --- Pydantic & SQLAlchemy ---
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel
from sqlalchemy import (
    Column, Date, Integer, String, SmallInteger,
    create_engine, text
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------

RUTA_JSON = Path("sorteos.json")


# ===========================================================================
# CONFIGURACIÓN: Pydantic Settings (lee .env automáticamente)
# ===========================================================================


class ConfigDB(BaseSettings):
    """
    Lee las variables de entorno desde el fichero .env.

    Pydantic BaseSettings busca el .env en el directorio de trabajo actual.
    Si la variable no existe en el entorno ni en .env, lanza ValidationError.

    La URL puede venir como 'postgres://' (heroku-style) o
    'postgresql+psycopg2://' (SQLAlchemy moderno). El validator normaliza
    automáticamente el prefijo.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    database_url: str

    @field_validator("database_url", mode="before")
    @classmethod
    def normalizar_url(cls, v: str) -> str:
        """
        SQLAlchemy ≥ 1.4 no acepta 'postgres://', solo 'postgresql://'.
        También añade el driver psycopg2 si no está especificado.
        """
        if v.startswith("postgres://"):
            v = v.replace("postgres://", "postgresql://", 1)
        if v.startswith("postgresql://") and "+psycopg2" not in v:
            v = v.replace("postgresql://", "postgresql+psycopg2://", 1)
        return v


# ===========================================================================
# MODELOS Pydantic: validación de datos antes de insertar en DB
# ===========================================================================


class SorteoSchema(BaseModel):
    """
    Esquema de validación para un registro de sorteo.
    Pydantic valida tipos y constraints antes de que el dato llegue a la DB.
    """

    tipo_sorteo: str
    fecha: date
    bolas: list[int]
    especiales: Optional[list[int]] = None

    @model_validator(mode="after")
    def validar_bolas(self) -> "SorteoSchema":
        """
        Valida integridad de las bolas antes de persistir en DB.

        Reglas:
            - Bolas principales: deben ser >= 1 (nunca 0 ni negativos).
            - Números especiales: deben ser >= 0. El 0 es un valor válido
              en algunos sorteos (p.ej. el Reintegro de Primitiva va de 0 a 9).
            - No se admiten duplicados dentro de las bolas principales.
        """
        if any(b < 1 for b in self.bolas):
            raise ValueError(
                "Las bolas principales deben ser >= 1 "
                f"(valores recibidos: {self.bolas})."
            )
        if len(self.bolas) != len(set(self.bolas)):
            raise ValueError("Las bolas principales no pueden repetirse.")
        if self.especiales is not None:
            if any(e < 0 for e in self.especiales):
                raise ValueError(
                    "Los números especiales no pueden ser negativos "
                    f"(valores recibidos: {self.especiales})."
                )
        return self


# ===========================================================================
# CAPA DB: SQLAlchemy ORM
# ===========================================================================


class Base(DeclarativeBase):
    """Base declarativa para todos los modelos ORM."""
    pass


class SorteoORM(Base):
    """
    Modelo ORM que mapea a la tabla 'sorteo' en PostgreSQL.

    Diseño de tabla (ver POSTGRESQL_DESIGN.md):
        - Usa ARRAY de SMALLINT para las bolas → flexibilidad y consulta ANY().
        - Una sola tabla para todos los sorteos (gordo, primitiva, euromillones).
        - UNIQUE(tipo_sorteo, fecha) evita duplicados en carga incremental.
    """

    __tablename__ = "sorteo"

    id = Column(Integer, primary_key=True, autoincrement=True)
    tipo_sorteo = Column(String(20), nullable=False, index=True)
    fecha = Column(Date, nullable=False, index=True)
    bolas = Column(ARRAY(SmallInteger), nullable=False)
    especiales = Column(ARRAY(SmallInteger), nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Sorteo tipo={self.tipo_sorteo!r} "
            f"fecha={self.fecha} bolas={self.bolas}>"
        )


def crear_engine(url: str):
    """
    Crea el engine de SQLAlchemy con pool de conexiones básico.

    Args:
        url: URL de conexión normalizada (postgresql+psycopg2://...).

    Returns:
        Engine de SQLAlchemy.
    """
    return create_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,   # Verifica conexión antes de usarla
        echo=False,           # True para ver SQL generado en debug
    )


def inicializar_db(engine) -> None:
    """
    Crea las tablas si no existen. Operación idempotente (checkfirst=True).

    Args:
        engine: Engine de SQLAlchemy conectado a PostgreSQL.
    """
    Base.metadata.create_all(engine, checkfirst=True)
    print("[DB] Tablas verificadas/creadas correctamente.")


def guardar_sorteos_en_db(
    df: pd.DataFrame,
    tipo_sorteo: str,
    columnas_bolas: list[str],
    columnas_especiales: list[str],
    session: Session,
) -> tuple[int, int]:
    """
    Inserta o actualiza los sorteos del DataFrame en PostgreSQL.

    Estrategia upsert:
        - Consulta los (tipo_sorteo, fecha) ya presentes en DB.
        - Solo inserta los registros nuevos (carga incremental).
        - Valida cada registro con SorteoSchema antes de insertar.

    Args:
        df: DataFrame limpio ordenado por fecha.
        tipo_sorteo: Nombre del sorteo ('primitiva', 'gordo', 'euromillones').
        columnas_bolas: Columnas de bolas principales.
        columnas_especiales: Columnas de números especiales.
        session: Sesión SQLAlchemy activa.

    Returns:
        Tupla (insertados, omitidos).
    """
    # Obtener fechas ya presentes para este tipo de sorteo
    fechas_existentes: set[date] = {
        row[0]
        for row in session.execute(
            text(
                "SELECT fecha FROM sorteo "
                "WHERE tipo_sorteo = :tipo"
            ),
            {"tipo": tipo_sorteo},
        )
    }

    insertados = 0
    omitidos = 0

    for _, fila in df.iterrows():
        fecha_sorteo = fila["Fecha"].date()

        if fecha_sorteo in fechas_existentes:
            omitidos += 1
            continue

        bolas = fila[columnas_bolas].dropna().astype(int).tolist()
        especiales = (
            fila[columnas_especiales].dropna().astype(int).tolist()
            if columnas_especiales
            else None
        )

        try:
            # Validación con Pydantic antes de tocar la DB
            schema = SorteoSchema(
                tipo_sorteo=tipo_sorteo,
                fecha=fecha_sorteo,
                bolas=bolas,
                especiales=especiales if especiales else None,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARN] Sorteo {fecha_sorteo} inválido, omitido: {exc}")
            omitidos += 1
            continue

        registro = SorteoORM(
            tipo_sorteo=schema.tipo_sorteo,
            fecha=schema.fecha,
            bolas=schema.bolas,
            especiales=schema.especiales,
        )
        session.add(registro)
        insertados += 1

    session.commit()
    print(
        f"[DB] '{tipo_sorteo}': {insertados} registros insertados, "
        f"{omitidos} omitidos (ya existían o inválidos)."
    )
    return insertados, omitidos

def cargar_df_desde_db(
    tipo_sorteo: str,
    columnas_bolas: list[str],
    columnas_especiales: list[str],
    session: Session,
) -> pd.DataFrame:
    """
    Carga el histórico completo de un sorteo desde PostgreSQL como DataFrame.

    Reconstruye el mismo formato de columnas que produce cargar_y_limpiar_datos()
    para que todas las funciones de análisis reciban el mismo tipo de datos
    independientemente de si el origen es CSV o DB.

    El array 'bolas' de la DB se desempaqueta en columnas separadas
    (Bola1, Bola2, ...) usando los nombres definidos en columnas_bolas.
    Lo mismo con los especiales.

    Args:
        tipo_sorteo: Nombre del sorteo ('primitiva', 'gordo', 'euromillones').
        columnas_bolas: Nombres de columna para las bolas principales.
        columnas_especiales: Nombres de columna para los números especiales.
        session: Sesión SQLAlchemy activa.

    Returns:
        DataFrame ordenado por fecha con las mismas columnas que el CSV limpio,
        o DataFrame vacío si no hay registros.
    """
    filas = session.execute(
        text(
            "SELECT fecha, bolas, especiales FROM sorteo "
            "WHERE tipo_sorteo = :tipo "
            "ORDER BY fecha ASC"
        ),
        {"tipo": tipo_sorteo},
    ).fetchall()

    if not filas:
        return pd.DataFrame()

    registros: list[dict] = []
    for fila in filas:
        registro: dict = {"Fecha": pd.to_datetime(fila[0])}

        # Desempaquetar array de bolas en columnas individuales
        bolas_db: list[int] = fila[1] or []
        for i, col in enumerate(columnas_bolas):
            registro[col] = bolas_db[i] if i < len(bolas_db) else None

        # Desempaquetar especiales si existen
        especiales_db: list[int] = fila[2] or []
        for i, col in enumerate(columnas_especiales):
            registro[col] = especiales_db[i] if i < len(especiales_db) else None

        registros.append(registro)

    df = pd.DataFrame(registros)

    # Asegurar tipos Int64 (nullable) igual que cargar_y_limpiar_datos
    for col in columnas_bolas + columnas_especiales:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df.sort_values("Fecha", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(
        f"[DB] '{tipo_sorteo}': {len(df)} registros cargados desde PostgreSQL."
    )
    return df


# ===========================================================================
# CAPA 1: INGESTA / IO
# ===========================================================================


def cargar_config_sorteos(ruta: Path = RUTA_JSON) -> list[dict]:
    """
    Lee y valida el fichero de configuración de sorteos.

    Args:
        ruta: Ruta al fichero JSON con la definición de cada sorteo.

    Returns:
        Lista de diccionarios, uno por sorteo configurado.

    Raises:
        FileNotFoundError: Si el fichero JSON no existe.
        ValueError: Si el JSON no contiene una lista válida.
    """
    import json

    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero de configuración: {ruta}"
        )
    with ruta.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("El fichero JSON debe contener una lista de sorteos.")
    return data


def cargar_y_limpiar_datos(config: dict) -> pd.DataFrame:
    """
    Carga el CSV de un sorteo, normaliza tipos y elimina filas inválidas.

    Pipeline:
        1. Lectura del CSV saltando la cabecera original.
        2. Conversión de 'Fecha' a datetime (formato dd/mm/yyyy).
        3. Conversión de columnas numéricas a Int64 (nullable integer).
        4. Eliminación de filas con bolas principales nulas.
        5. Ordenación cronológica ascendente.

    Args:
        config: Dict con claves 'fichero', 'columnas', 'numeros',
                'numeros_especiales'.

    Returns:
        DataFrame limpio ordenado por fecha, o DataFrame vacío si hay error.
    """
    ruta_csv = Path(config["fichero"])
    print(f"\n{'='*60}")
    print(f"  Cargando sorteo: {config['sorteo'].upper()}")
    print(f"  Fichero: {ruta_csv}")
    print(f"{'='*60}")

    if not ruta_csv.exists():
        print(f"[ERROR] Fichero '{ruta_csv}' no encontrado.")
        return pd.DataFrame()

    df = pd.read_csv(ruta_csv, sep=",", skiprows=1, names=config["columnas"])
    df["Fecha"] = pd.to_datetime(df["Fecha"], format="%d/%m/%Y", errors="coerce")
    df.dropna(subset=["Fecha"], inplace=True)

    cols_num: list[str] = config["numeros"] + config["numeros_especiales"]
    for col in cols_num:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df.dropna(subset=config["numeros"], inplace=True)
    df.sort_values(by="Fecha", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"[OK] {len(df)} sorteos cargados tras limpieza.")
    return df


# ===========================================================================
# CAPA 2: ANÁLISIS CLÁSICO
# ===========================================================================


def analizar_frecuencia_numeros(
    df: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """
    Calcula la frecuencia absoluta y relativa de cada número.

    Matemática:
        P(n) = frecuencia(n) / Σ_i frecuencia(i)

    Args:
        df: DataFrame con el histórico de sorteos.
        columnas: Columnas de las bolas a analizar.

    Returns:
        DataFrame con columnas [Numero, Frecuencia, Probabilidad (%)].
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

    print("\n--- FRECUENCIA DE NÚMEROS ---")
    print(f"Basado en {len(df)} sorteos.")
    print(resultado.to_string(index=False))
    return resultado


def analizar_diferencia_fechas(
    df: pd.DataFrame, columnas: list[str]
) -> pd.DataFrame:
    """
    Estadísticos de los intervalos (días) entre apariciones de cada número.

    Para cada número n:
        - Filtra los sorteos donde n apareció.
        - Calcula la diferencia en días entre apariciones consecutivas.
        - Extrae media, mínimo y máximo de esas diferencias.

    Args:
        df: DataFrame con columna 'Fecha' y columnas de bolas.
        columnas: Columnas de bolas a analizar.

    Returns:
        DataFrame con [Numero, Dias Promedio, Dias Min, Dias Max,
        Ultima Aparicion].
    """
    print("\n--- PATRONES TEMPORALES ---")
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
        print("No hay suficientes datos para analizar los patrones de fechas.")
        return pd.DataFrame()

    resultado = pd.DataFrame(filas)
    print(resultado.drop(columns=["Ultima Aparicion"]).to_string(index=False))
    return resultado


def analizar_combinaciones(
    df: pd.DataFrame, columnas: list[str], tamano_grupo: int
) -> None:
    """
    Combinaciones de tamaño k más frecuentes en el histórico.

    Matemática: C(n, k) = n! / (k! * (n-k)!)
    Se generan todas las combinaciones de cada sorteo y se acumulan en Counter.

    Args:
        df: DataFrame del histórico.
        columnas: Columnas de bolas.
        tamano_grupo: Tamaño k (2=pares, 3=tríos, ...).
    """
    print(f"\n--- COMBINACIONES MÁS FRECUENTES (k={tamano_grupo}) ---")
    contador: Counter = Counter()

    for _, fila in df.iterrows():
        bolas = sorted(fila[columnas].dropna().astype(int).tolist())
        if len(bolas) >= tamano_grupo:
            contador.update(combinations(bolas, tamano_grupo))

    if not contador:
        print("No se encontraron combinaciones.")
        return

    for combo, freq in contador.most_common(15):
        print(f"  {combo}: {freq} veces")


def calcular_indice_tendencia(
    df_freq: pd.DataFrame,
    df_fechas: pd.DataFrame,
    fecha_referencia: date,
    peso_urgencia: float = 0.7,
    peso_frecuencia: float = 0.3,
) -> pd.DataFrame:
    """
    Índice de TENDENCIA ESTADÍSTICA: combina urgencia temporal y frecuencia.
    NO implica predicción de sorteos reales.

    Fórmula:
        urgencia(n)  = dias_sin_salir(n) / dias_promedio(n)
        freq_norm(n) = (freq(n) - min_freq) / (max_freq - min_freq)
        indice(n)    = urgencia(n) * peso_urgencia
                      + freq_norm(n) * peso_frecuencia

    Args:
        df_freq: Resultado de analizar_frecuencia_numeros().
        df_fechas: Resultado de analizar_diferencia_fechas().
        fecha_referencia: Fecha para calcular días sin salir.
        peso_urgencia: Peso del factor temporal [0, 1].
        peso_frecuencia: Peso del factor frecuencia [0, 1].

    Returns:
        DataFrame ordenado por índice descendente.
    """
    print(f"\n--- ÍNDICE DE TENDENCIA ESTADÍSTICA ({fecha_referencia}) ---")
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

    cols = ["Numero", "Indice", "Urgencia", "Dias Sin Salir",
            "Dias Promedio", "Frecuencia"]
    print(df[cols].head(15).to_string(index=False))
    return df


# ===========================================================================
# CAPA 3: CADENAS DE MARKOV
# ===========================================================================


class MarkovLoteria:
    """
    Cadena de Markov de primer orden sobre la secuencia de sorteos.

    Cada sorteo → ESTADO DISCRETO.
    La matriz T[i][j] = P(S_{t+1}=j | S_t=i) se aprende del histórico.

    Propiedad de Markov: P(S_{t+1} | S_0..S_t) = P(S_{t+1} | S_t)

    Estados disponibles:
        - 'paridad'   → {'par_dom', 'impar_dom', 'empate'}
        - 'decenio'   → {'0-9', '10-19', ...} según mediana de las bolas
        - 'zona_suma' → {'bajo', 'medio', 'alto'} por terciles de suma
    """

    TIPOS_ESTADO = ("paridad", "decenio", "zona_suma")

    def __init__(self, tipo_estado: str = "zona_suma") -> None:
        """
        Args:
            tipo_estado: Uno de 'paridad', 'decenio' o 'zona_suma'.

        Raises:
            ValueError: Si el tipo no es válido.
        """
        if tipo_estado not in self.TIPOS_ESTADO:
            raise ValueError(
                f"tipo_estado debe ser uno de {self.TIPOS_ESTADO}"
            )
        self.tipo_estado = tipo_estado
        self.matriz_transicion: pd.DataFrame = pd.DataFrame()
        self.estados_secuencia: list[str] = []
        self._terciles: Optional[tuple[float, float]] = None

    def definir_estado(self, bolas: list[int]) -> str:
        """
        Mapea un sorteo a su estado discreto según el tipo_estado configurado.

        Args:
            bolas: Lista de números del sorteo.

        Returns:
            Cadena que representa el estado discreto.

        Raises:
            RuntimeError: Si se llama con tipo_estado='zona_suma' sin que
                         los terciles hayan sido precalculados.
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
            base = int(float(np.median(bolas)) // 10) * 10
            return f"{base}-{base + 9}"

        # zona_suma
        if self._terciles is None:
            raise RuntimeError(
                "Los terciles no han sido calculados. "
                "Llama primero a construir_matriz_transicion()."
            )
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
        Construye la matriz de transición estocástica T.

        Proceso:
            1. Mapear cada sorteo a un estado discreto.
            2. Contar transiciones estado_i → estado_j.
            3. Normalizar filas para obtener P(j|i).

        Args:
            df: DataFrame ordenado cronológicamente.
            columnas: Columnas de las bolas principales.

        Returns:
            DataFrame con la matriz T (índice=origen, cols=destino).
        """
        # Precalcular terciles si el tipo es 'zona_suma'
        if self.tipo_estado == "zona_suma":
            sumas = df[columnas].apply(
                lambda r: r.dropna().astype(int).sum(), axis=1
            )
            q33, q66 = float(np.percentile(sumas, 33)), float(
                np.percentile(sumas, 66)
            )
            self._terciles = (q33, q66)

        # Generar secuencia de estados
        estados: list[str] = []
        for _, fila in df.iterrows():
            bolas = fila[columnas].dropna().astype(int).tolist()
            if bolas:
                estados.append(self.definir_estado(bolas))

        self.estados_secuencia = estados

        # Contar transiciones
        contador: Counter = Counter(zip(estados[:-1], estados[1:]))
        estados_unicos = sorted(set(estados))
        matriz = pd.DataFrame(0, index=estados_unicos, columns=estados_unicos)

        for (origen, destino), cuenta in contador.items():
            matriz.loc[origen, destino] = cuenta

        # Normalizar filas → probabilidades de transición
        totales = matriz.sum(axis=1)
        self.matriz_transicion = matriz.div(totales, axis=0).fillna(0)

        print(
            f"\n--- MATRIZ DE TRANSICIÓN MARKOV "
            f"(estado: {self.tipo_estado}) ---"
        )
        print(self.matriz_transicion.round(3).to_string())
        return self.matriz_transicion

    def probabilidad_siguiente_estado(self, estado_actual: str) -> pd.Series:
        """
        Distribución de probabilidad del siguiente estado dado el actual.

        Matemática: P(S_{t+1} | S_t = estado_actual) = fila de T.

        Args:
            estado_actual: Estado del sorteo más reciente.

        Returns:
            Series con P(S_{t+1}=j | S_t=estado_actual) para cada estado j.

        Raises:
            RuntimeError: Si no se ha construido la matriz.
            ValueError: Si el estado no existe en la matriz.
        """
        if self.matriz_transicion.empty:
            raise RuntimeError("Llama primero a construir_matriz_transicion().")
        if estado_actual not in self.matriz_transicion.index:
            raise ValueError(
                f"Estado '{estado_actual}' no encontrado. "
                f"Disponibles: {list(self.matriz_transicion.index)}"
            )

        dist = self.matriz_transicion.loc[estado_actual].sort_values(
            ascending=False
        )
        print(f"\n  Desde '{estado_actual}', tendencias del siguiente estado:")
        for estado, prob in dist.items():
            print(f"    → '{estado}': {prob:.1%}")
        return dist


# ===========================================================================
# CAPA 4: DEEP LEARNING (PyTorch)
# ===========================================================================


def preparar_secuencias_lstm(
    df: pd.DataFrame,
    columnas: list[str],
    max_numero: int,
    ventana: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Transforma el histórico en pares (entrada, salida) para el LSTM.

    Codificación one-hot por sorteo:
        vec[i-1] = 1.0 si el número i salió, 0.0 si no.

    Ventanas deslizantes:
        x_seq[t] = [vec_{t-w+1}, ..., vec_t]  shape: (ventana, max_numero)
        y_seq[t] = vec_{t+1}                   shape: (max_numero,)

    Args:
        df: DataFrame ordenado cronológicamente.
        columnas: Columnas de bolas principales.
        max_numero: Número máximo posible (p.ej. 49 para Primitiva).
        ventana: Número de sorteos pasados por muestra.

    Returns:
        (x_tensor, y_tensor):
            x_tensor: float32 shape (N, ventana, max_numero)
            y_tensor: float32 shape (N, max_numero)
    """
    vectores: list[np.ndarray] = []
    for _, fila in df.iterrows():
        bolas = fila[columnas].dropna().astype(int).tolist()
        vec = np.zeros(max_numero, dtype=np.float32)
        for bola in bolas:
            if 1 <= bola <= max_numero:
                vec[bola - 1] = 1.0
        vectores.append(vec)

    x_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    for i in range(ventana, len(vectores)):
        x_list.append(np.stack(vectores[i - ventana: i]))
        y_list.append(vectores[i])
    if not x_list:
        return (
            torch.empty(0, ventana, max_numero, dtype=torch.float32),
            torch.empty(0, max_numero, dtype=torch.float32),
        )
    x_arr = np.stack(x_list)   # (N, ventana, max_numero)
    y_arr = np.stack(y_list)   # (N, max_numero)
    return (
        torch.tensor(x_arr, dtype=torch.float32),
        torch.tensor(y_arr, dtype=torch.float32),
    )


class HiperparametrosLSTM:
    """Agrupa los hiperparámetros de entrenamiento para reducir la firma."""

    def __init__(
        self,
        epochs: int = 50,
        batch_size: int = 16,
        lr: float = 0.001,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr


class LSTMLoteria(nn.Module):
    """
    LSTM para análisis de tendencias en series temporales de sorteos.

    AVISO: Identifica correlaciones estadísticas con fines educativos.
    NO predice sorteos reales (eventos independientes).

    Arquitectura:
        Entrada → LSTM (num_capas) → FC → ReLU → Dropout → FC → Sigmoid

        - Entrada: secuencia de vectores one-hot de sorteos pasados.
        - Salida: vector en [0, 1] con una probabilidad por número.
        - Sigmoid (no Softmax): cada número es independiente del resto.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_capas: int = 2,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            input_size:  Dimensión de entrada (= max_numero del sorteo).
            hidden_size: Dimensión del estado oculto h_t del LSTM.
            num_capas:   Capas LSTM apiladas.
            dropout:     Dropout entre capas (no se aplica en la última).
        """
        super().__init__()
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
        Propagación hacia adelante.

        Args:
            x_input: (batch_size, seq_len, input_size)

        Returns:
            Tensor (batch_size, input_size) con probabilidades en [0, 1].

        Flujo:
            x_input → LSTM → h_t (última) → FC → ReLU → Dropout
                    → FC salida → Sigmoid
        """
        lstm_out, _ = self.lstm(x_input)
        last_out = lstm_out[:, -1, :]                   # (batch, hidden)
        hidden = self.relu(self.fc_hidden(last_out))
        hidden = self.dropout(hidden)
        return self.sigmoid(self.fc_output(hidden))


def entrenar_lstm(
    modelo: LSTMLoteria,
    x_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    hp: Optional[HiperparametrosLSTM] = None,
) -> None:
    """
    Entrena el LSTM con Binary Cross Entropy Loss.

    BCE = -Σ [y_i * log(p_i) + (1 - y_i) * log(1 - p_i)]
    y_i ∈ {0, 1}, p_i ∈ [0, 1] es la salida del modelo.

    Args:
        modelo: Instancia de LSTMLoteria.
        x_tensor: Tensor de entrada (N, ventana, max_numero).
        y_tensor: Tensor de salida (N, max_numero).
        hp: Hiperparámetros (usa valores por defecto si es None).
    """
    if hp is None:
        hp = HiperparametrosLSTM()

    dataset = TensorDataset(x_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(modelo.parameters(), lr=hp.lr)

    print(f"\n{'='*60}")
    print(f"  ENTRENAMIENTO LSTM — {hp.epochs} épocas")
    print(f"{'='*60}")

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

        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Época {epoch:3d}/{hp.epochs} | Loss: {avg_loss:.4f}")

    print("[OK] Entrenamiento completado.\n")


def predecir_tendencias_lstm(
    modelo: LSTMLoteria,
    ultima_secuencia: torch.Tensor,
    max_numero: int,
    top_k: int = 15,
) -> list[tuple[int, float]]:
    """
    Genera probabilidades estadísticas del siguiente sorteo según el LSTM.

    AVISO: El modelo identifica patrones en el histórico. NO tiene capacidad
    predictiva real sobre sorteos independientes.

    Args:
        modelo: LSTMLoteria entrenado.
        ultima_secuencia: Tensor (1, ventana, max_numero).
        max_numero: Número máximo del sorteo.
        top_k: Números top a retornar.

    Returns:
        Lista de (numero, probabilidad) ordenada por probabilidad desc.
    """
    modelo.eval()
    with torch.no_grad():
        probs = modelo(ultima_secuencia).squeeze(0).numpy()
    resultados = [(i + 1, float(probs[i])) for i in range(max_numero)]
    resultados.sort(key=lambda par: par[1], reverse=True)
    return resultados[:top_k]


# ===========================================================================
# CAPA 5: ORQUESTADOR PRINCIPAL
# ===========================================================================


def _ejecutar_analisis_sorteo(
    df: pd.DataFrame,
    config: dict,
    fecha_proximo_sorteo: date,
) -> None:
    """
    Ejecuta todas las fases de análisis para un único sorteo.
    Función auxiliar para mantener main() limpio y dentro de límites de linter.

    Args:
        df: DataFrame limpio del sorteo.
        config: Configuración del sorteo (columnas, tipo, etc.).
        fecha_proximo_sorteo: Fecha de referencia para el índice de tendencia.
    """
    cols_principales: list[str] = config["numeros"]
    cols_especiales: list[str] = config["numeros_especiales"]

    # --- ANÁLISIS CLÁSICO ---
    df_freq = analizar_frecuencia_numeros(df, cols_principales)
    df_fechas = analizar_diferencia_fechas(df, cols_principales)
    analizar_combinaciones(df, cols_principales, tamano_grupo=2)
    analizar_combinaciones(df, cols_principales, tamano_grupo=3)

    if not df_freq.empty and not df_fechas.empty:
        calcular_indice_tendencia(df_freq, df_fechas, fecha_proximo_sorteo)

    # --- CADENAS DE MARKOV ---
    print(f"\n{'='*60}")
    print("  ANÁLISIS DE CADENAS DE MARKOV")
    print(f"{'='*60}")

    for tipo in MarkovLoteria.TIPOS_ESTADO:
        markov = MarkovLoteria(tipo_estado=tipo)
        markov.construir_matriz_transicion(df, cols_principales)
        bolas_ult = (
            df[cols_principales].dropna().iloc[-1].astype(int).tolist()
        )
        estado_actual = markov.definir_estado(bolas_ult)
        markov.probabilidad_siguiente_estado(estado_actual)

    # --- DEEP LEARNING: LSTM ---
    print(f"\n{'='*60}")
    print("  ANÁLISIS LSTM (SERIES TEMPORALES)")
    print(f"{'='*60}")

    max_num = int(
        pd.concat([df[c] for c in cols_principales]).dropna().max()
    )
    ventana = 10
    x_tensor, y_tensor = preparar_secuencias_lstm(
        df, cols_principales, max_numero=max_num, ventana=ventana
    )

    if x_tensor.shape[0] < 20:
        print("[SKIP] Datos insuficientes para LSTM (< 20 muestras).")
    else:
        modelo = LSTMLoteria(input_size=max_num)
        entrenar_lstm(modelo, x_tensor, y_tensor)

        # Guardar el modelo en la carpeta de modelos entrenados de la app Django
        dir_modelos = Path("analytics/modelos_entrenados")
        dir_modelos.mkdir(parents=True, exist_ok=True)
        ruta_modelo = dir_modelos / f"lstm_{config['sorteo']}.pt"
        torch.save(modelo.state_dict(), ruta_modelo)
        print(f"[OK] Modelo LSTM guardado en: {ruta_modelo}")

        ultima_seq = x_tensor[-1].unsqueeze(0)
        tendencias = predecir_tendencias_lstm(modelo, ultima_seq, max_num)
        print("Top 15 números por tendencia estadística LSTM:")
        for num, prob in tendencias:
            progreso = "█" * int(prob * 20)
            print(f"  Número {num:2d}: {prob:.4f}  {progreso}")

    # --- NÚMEROS ESPECIALES ---
    if cols_especiales:
        print(f"\n{'='*60}")
        print("  ANÁLISIS DE NÚMEROS ESPECIALES")
        print(f"{'='*60}")
        analizar_frecuencia_numeros(df, cols_especiales)


def main() -> None:
    """
    Pipeline principal: carga config, conecta a DB y ejecuta todos los análisis.

    Flujo por sorteo:
        1. Leer ConfigDB desde .env (Pydantic Settings).
        2. Conectar a PostgreSQL y crear tablas si no existen (SQLAlchemy).
        3. Para cada sorteo en sorteos.json:
           a. Cargar CSV y guardar en DB solo los registros nuevos.
              En ejecuciones posteriores, si no hay registros nuevos en el CSV
              respecto a la DB, no se inserta nada.
           b. Cargar el histórico COMPLETO desde la DB (fuente de verdad).
           c. Ejecutar todos los análisis estadísticos sobre los datos de DB.

    Separar "origen de datos para inserción" (CSV) de "origen de datos para
    análisis" (DB) garantiza que los análisis siempre reflejan el estado
    completo y persistido, no solo el contenido del CSV local.
    """
    cfg = ConfigDB()
    print("[CONFIG] Conectando a la base de datos...")

    engine = crear_engine(cfg.database_url)
    inicializar_db(engine)
    SessionLocal = sessionmaker(bind=engine)

    configs = cargar_config_sorteos()

    hoy = date.today()
    dias_hasta_domingo = (6 - hoy.weekday() + 7) % 7 or 7
    fecha_proximo_sorteo = hoy + timedelta(days=dias_hasta_domingo)

    for config in configs:
        tipo = config["sorteo"]
        cols_bolas: list[str] = config["numeros"]
        cols_especiales: list[str] = config["numeros_especiales"]

        # --- PASO 1: Cargar CSV e insertar solo registros nuevos en DB ---
        df_csv = cargar_y_limpiar_datos(config)

        with SessionLocal() as session:
            if not df_csv.empty:
                print(f"\n[DB] Sincronizando '{tipo}' con PostgreSQL...")
                insertados, omitidos = guardar_sorteos_en_db(
                    df=df_csv,
                    tipo_sorteo=tipo,
                    columnas_bolas=cols_bolas,
                    columnas_especiales=cols_especiales,
                    session=session,
                )
                if insertados == 0:
                    print(
                        f"[DB] Sin registros nuevos en el CSV para '{tipo}'."
                    )

            # --- PASO 2: Cargar histórico completo desde DB ---
            print(f"\n[DB] Cargando histórico de '{tipo}' desde PostgreSQL...")
            df = cargar_df_desde_db(tipo, cols_bolas, cols_especiales, session)

        if df.empty:
            print(f"[SKIP] Sin datos en DB para '{tipo}'. Ejecuta primero con CSV.\n")
            continue

        # --- PASO 3: Análisis sobre datos de la DB ---
        _ejecutar_analisis_sorteo(df, config, fecha_proximo_sorteo)
        print(f"\n{'='*60}\n")


if __name__ == "__main__":
    main()
