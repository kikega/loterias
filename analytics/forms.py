from datetime import date
from typing import Any

from django import forms

from .ml_services import config_por_tipo, tipos_disponibles


class SorteoForm(forms.Form):
    """
    Formulario base para la inserción de un sorteo.
    Define los campos fijos y carga dinámicamente las opciones de tipos de sorteos disponibles.
    """
    tipo_sorteo = forms.ChoiceField(
        choices=[],
        label="Tipo de sorteo",
        widget=forms.Select(
            attrs={
                "class": "w-full rounded border px-3 py-2",
                "onchange": "window.location.href='?tipo=' + this.value",
            }
        ),
    )
    fecha = forms.DateField(
        label="Fecha del sorteo",
        initial=date.today,
        widget=forms.DateInput(
            attrs={
                "type": "date",
                "class": "w-full rounded border px-3 py-2",
            }
        ),
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Constructor del formulario. Carga de manera dinámica los tipos de sorteo activos.

        Args:
            *args: Argumentos posicionales para el inicializador de Django forms.
            **kwargs: Argumentos por nombre (p. ej. initial, data) para el inicializador de Django.
        """
        super().__init__(*args, **kwargs)
        self.fields["tipo_sorteo"].choices = [
            (t, t.capitalize()) for t in tipos_disponibles()
        ]

    def clean(self) -> dict[str, Any]:
        """
        Valida que los números de bolas principales y especiales estén dentro de rangos coherentes.

        Returns:
            dict[str, Any]: Diccionario con los datos saneados y validados.
        """
        cleaned = super().clean()
        tipo = cleaned.get("tipo_sorteo")
        if tipo:
            cfg = config_por_tipo(tipo)
            num_bolas = len(cfg["numeros"])
            num_especiales = len(cfg["numeros_especiales"])
            for i in range(1, num_bolas + 1):
                field_name = f"bola_{i}"
                val = self.cleaned_data.get(field_name)
                if val is None or val < 1:
                    self.add_error(
                        field_name,
                        f"La bola {i} debe ser un número >= 1.",
                    )
            for i in range(1, num_especiales + 1):
                field_name = f"especial_{i}"
                val = self.cleaned_data.get(field_name)
                if val is None or val < 0:
                    self.add_error(
                        field_name,
                        f"El especial {i} debe ser un número >= 0.",
                    )
        return cleaned


def construir_formulario_sorteo(tipo: str) -> type[forms.Form]:
    """
    Función factoría que construye dinámicamente una subclase de SorteoForm
    configurada con el número de bolas y números especiales correspondientes al tipo de sorteo.

    Args:
        tipo (str): Identificador del tipo de sorteo (p. ej. 'primitiva', 'gordo').

    Returns:
        type[forms.Form]: Clase del formulario adaptada dinámicamente con los campos correspondientes.
    """
    cfg = config_por_tipo(tipo)
    num_bolas = len(cfg["numeros"])
    num_especiales = len(cfg["numeros_especiales"])
    base_fields: dict[str, forms.Field] = {
        f"bola_{i}": forms.IntegerField(
            label=f"Bola {i}",
            min_value=1,
            widget=forms.NumberInput(
                attrs={
                    "class": "w-full rounded border px-3 py-2 text-center",
                    "placeholder": f"Bola {i}",
                }
            ),
        )
        for i in range(1, num_bolas + 1)
    }
    for i in range(1, num_especiales + 1):
        label = cfg["numeros_especiales"][i - 1]
        base_fields[f"especial_{i}"] = forms.IntegerField(
            label=label,
            min_value=0,
            required=any(
                cfg["numeros_especiales"][i - 1] != "Clave"
                for _ in [1]
            ),
            widget=forms.NumberInput(
                attrs={
                    "class": "w-full rounded border px-3 py-2 text-center",
                    "placeholder": label,
                }
            ),
        )
    return type("SorteoDinamicoForm", (SorteoForm,), base_fields)
