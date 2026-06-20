from django.contrib import admin

from .models import Sorteo


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
