import json
from typing import Any

import pandas as pd
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render

from .forms import construir_formulario_sorteo
from .ml_services import (
    config_por_tipo,
    df_desde_orm,
    ejecutar_analisis,
    tipos_disponibles,
)
from .models import Sorteo


def _limpiar_nombre_col(name: str) -> str:
    """
    Sanea el nombre de una columna eliminando caracteres incompatibles con
    las plantillas HTML o el parseo (p. ej., espacios, paréntesis, símbolos de porcentaje).

    Args:
        name (str): Nombre de columna original.

    Returns:
        str: Nombre saneado y compatible.
    """
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")


def _serializar_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Convierte un DataFrame de pandas a una lista de diccionarios JSON serializable,
    limpiando previamente los nombres de las columnas.

    Args:
        df (pd.DataFrame): DataFrame a serializar.

    Returns:
        list[dict[str, Any]]: Lista de registros en formato de diccionario.
    """
    if df is None or df.empty:
        return []
    df = df.rename(columns=_limpiar_nombre_col)
    return json.loads(df.to_json(orient="records", force_ascii=False))


def _serializar_markov(resultado: Any) -> dict[str, Any]:
    """
    Serializa los datos de las cadenas de Markov contenidos en el resultado
    del análisis para que sean legibles y estructurados en formato JSON para la plantilla.

    Args:
        resultado (ResultadoAnalisis): Objeto con los resultados de las cadenas de Markov.

    Returns:
        dict[str, Any]: Estructura serializada con matrices y probabilidades de Markov.
    """
    data = {}
    for tipo, info in resultado.markov.items():
        matriz = info["matriz"]
        data[tipo] = {
            "estado_actual": info["estado_actual"],
            "matriz": json.loads(matriz.to_json(orient="split"))
            if not matriz.empty
            else None,
            "distribucion": (
                info["distribucion"].to_dict() if not info["distribucion"].empty else {}
            ),
        }
    return data


@login_required
def dashboard(request: HttpRequest) -> HttpResponse:
    """
    Vista que sirve el shell base del Dashboard. Esta página carga de forma
    instantánea y delega la visualización del análisis a HTMX mediante una llamada diferida.

    Args:
        request (HttpRequest): Solicitud HTTP de Django.

    Returns:
        HttpResponse: Renderizado de la plantilla base del dashboard.
    """
    tipo = request.GET.get("tipo", "primitiva")
    if tipo not in tipos_disponibles():
        tipo = "primitiva"

    # Esta vista ahora es instantánea: sirve la maqueta base con el cargador/spinner
    ctx = {
        "tipos": tipos_disponibles(),
        "tipo_activo": tipo,
    }
    return render(request, "analytics/dashboard.html", ctx)


@login_required
def dashboard_content(request: HttpRequest) -> HttpResponse:
    """
    Vista asíncrona (invocada por HTMX) que ejecuta el análisis matemático de lotería
    sobre la base de datos y renderiza el contenido detallado del panel estadístico.

    Args:
        request (HttpRequest): Solicitud HTTP de Django.

    Returns:
        HttpResponse: HTML parcial con los gráficos, tablas e indicadores analíticos.
    """
    tipo = request.GET.get("tipo", "primitiva")
    if tipo not in tipos_disponibles():
        tipo = "primitiva"

    resultado = ejecutar_analisis(tipo, entrenar=False)

    lstm_top_formatted = [
        {
            "numero": n,
            "probabilidad": round(p, 4),
            "pct": f"{p * 100:.2f}"
        }
        for n, p in resultado.lstm_top
    ]

    ultimos_sorteos = Sorteo.objects.filter(tipo_sorteo=tipo).order_by("-fecha")[:10]

    ctx = {
        "tipo_activo": tipo,
        "total_sorteos": resultado.total_sorteos,
        "frecuencias": _serializar_df(resultado.frecuencias),
        "patrones_temporales": _serializar_df(resultado.patrones_temporales),
        "indice_tendencia": _serializar_df(resultado.indice_tendencia),
        "combinaciones_pares": resultado.combinaciones_pares,
        "combinaciones_trios": resultado.combinaciones_trios,
        "markov": _serializar_markov(resultado),
        "lstm_top": lstm_top_formatted,
        "frecuencias_especiales": _serializar_df(
            getattr(resultado, "frecuencias_especiales", pd.DataFrame())
        ),
        "ultimos_sorteos": sorted(list(ultimos_sorteos), key=lambda x: x.fecha, reverse=True),
    }
    return render(request, "analytics/dashboard_content.html", ctx)


@login_required
def dashboard_json(request: HttpRequest) -> JsonResponse:
    """
    API JSON que expone los datos básicos del análisis de loterías (frecuencia,
    tendencias y predicciones de red neuronal) para posibles integraciones de cliente.

    Args:
        request (HttpRequest): Solicitud HTTP de Django.

    Returns:
        JsonResponse: Datos analíticos serializados en formato JSON.
    """
    tipo = request.GET.get("tipo", "primitiva")
    if tipo not in tipos_disponibles():
        tipo = "primitiva"
    resultado = ejecutar_analisis(tipo, entrenar=False)
    data = {
        "tipo": tipo,
        "total_sorteos": resultado.total_sorteos,
        "frecuencias": _serializar_df(resultado.frecuencias),
        "indice_tendencia": _serializar_df(resultado.indice_tendencia),
        "lstm_top": [{"numero": n, "probabilidad": round(p, 4), "porcentaje": round(p * 100, 2)} for n, p in resultado.lstm_top],
    }
    return JsonResponse(data)


@login_required
def insertar_sorteo(request: HttpRequest) -> HttpResponse:
    """
    Vista para introducir manualmente nuevos resultados de sorteos a la base de datos.
    Genera el formulario adaptado al tipo de sorteo y lista los 10 últimos registros en el lateral.

    Args:
        request (HttpRequest): Solicitud HTTP de Django.

    Returns:
        HttpResponse: Página de inserción con formulario e historial lateral.
    """
    tipo = request.GET.get("tipo") or request.POST.get("tipo_sorteo", "primitiva")
    if tipo not in tipos_disponibles():
        tipo = "primitiva"

    if request.method == "POST":
        FormClass = construir_formulario_sorteo(tipo)
        form = FormClass(request.POST)
        if form.is_valid():
            cfg = config_por_tipo(tipo)
            data = form.cleaned_data
            num_bolas = len(cfg["numeros"])
            num_especiales = len(cfg["numeros_especiales"])
            bolas = [data[f"bola_{i}"] for i in range(1, num_bolas + 1)]
            especiales = (
                [data[f"especial_{i}"] for i in range(1, num_especiales + 1)]
                if num_especiales > 0
                else None
            )
            Sorteo.objects.create(
                tipo_sorteo=tipo,
                fecha=data["fecha"],
                bolas=bolas,
                especiales=especiales,
            )
            return redirect(f"/?tipo={tipo}")
    else:
        FormClass = construir_formulario_sorteo(tipo)
        form = FormClass(initial={"tipo_sorteo": tipo})

    cfg = config_por_tipo(tipo)
    tiene_especiales = len(cfg["numeros_especiales"]) > 0
    ultimos_sorteos = Sorteo.objects.filter(tipo_sorteo=tipo).order_by("-fecha")[:10]
    return render(request, "analytics/insertar.html", {
        "form": form,
        "tipos": tipos_disponibles(),
        "tipo_activo": tipo,
        "tiene_especiales": tiene_especiales,
        "ultimos_sorteos": sorted(list(ultimos_sorteos), key=lambda x: x.fecha, reverse=True),
    })


def login_view(request: HttpRequest) -> HttpResponse:
    """
    Vista de autenticación de usuarios. Valida credenciales e inicia sesión redirigiendo
    al panel de control o a la página solicitada de origen.

    Args:
        request (HttpRequest): Solicitud HTTP de Django.

    Returns:
        HttpResponse: Pantalla de inicio de sesión o redirección tras login correcto.
    """
    if request.user.is_authenticated:
        return redirect("dashboard")

    error = None
    if request.method == "POST":
        usuario = request.POST.get("username")
        clave = request.POST.get("password")
        user = authenticate(request, username=usuario, password=clave)
        if user is not None:
            login(request, user)
            next_url = request.GET.get("next", "dashboard")
            return redirect(next_url)
        else:
            error = "Usuario o contraseña incorrectos."

    return render(request, "analytics/login.html", {"error": error})


def logout_view(request: HttpRequest) -> HttpResponse:
    """
    Vista para cerrar la sesión activa del usuario y redirigir a la pantalla de login.

    Args:
        request (HttpRequest): Solicitud HTTP de Django.

    Returns:
        HttpResponse: Redirección al panel de inicio de sesión.
    """
    logout(request)
    return redirect("login")
