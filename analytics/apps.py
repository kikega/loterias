from django.apps import AppConfig


class AnalyticsConfig(AppConfig):
    """
    Configuración de la aplicación 'analytics' de Django.
    Se encarga de la inicialización de la app y la creación del directorio de modelos entrenados.
    """
    default_auto_field = "django.db.models.BigAutoField"
    name = "analytics"
    verbose_name = "Analítica de Loterías"

    def ready(self) -> None:
        """
        Código a ejecutar en la inicialización de la aplicación.
        Asegura que el directorio para guardar los modelos de machine learning esté disponible.
        """
        from .ml_services import RUTA_MODELOS

        RUTA_MODELOS.mkdir(parents=True, exist_ok=True)
