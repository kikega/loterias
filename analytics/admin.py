from django.contrib import admin

from .models import Sorteo, PrediccionSemanal, CombinacionPredicha


@admin.register(Sorteo)
class SorteoAdmin(admin.ModelAdmin):
    """
    Configuración personalizada del panel de administración de Django para el modelo Sorteo.
    Permite filtrar, buscar e inspeccionar los sorteos guardados por fecha y tipo.
    """
    list_display = ("tipo_sorteo", "fecha", "bolas", "especiales")
    list_filter = ("tipo_sorteo",)
    search_fields = ("tipo_sorteo",)
    date_hierarchy = "fecha"


class CombinacionPredichaInline(admin.TabularInline):
    model = CombinacionPredicha
    extra = 0
    fields = ("orden", "estrategia", "bolas", "especiales", "procesado", "aciertos_por_sorteo")
    readonly_fields = ("aciertos_por_sorteo",)


@admin.register(PrediccionSemanal)
class PrediccionSemanalAdmin(admin.ModelAdmin):
    list_display = ("tipo_sorteo", "anio", "semana", "fecha_creacion")
    list_filter = ("tipo_sorteo", "anio")
    search_fields = ("tipo_sorteo",)
    inlines = [CombinacionPredichaInline]

