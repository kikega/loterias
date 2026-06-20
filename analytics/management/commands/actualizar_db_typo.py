from django.core.management.base import BaseCommand
from analytics.models import Sorteo


class Command(BaseCommand):
    """
    Comando de gestión de Django para corregir errores de tipografía
    en los nombres de sorteo almacenados en la base de datos (p. ej. 'euromilones' -> 'euromillones').
    """
    help = "Corrige el tipo_sorteo de 'euromilones' a 'euromillones' en la base de datos."

    def handle(self, *args, **options) -> None:
        """
        Busca registros con error de tipografía en el campo tipo_sorteo y los corrige.

        Args:
            *args: Argumentos posicionales adicionales.
            **options: Opciones pasadas al comando.
        """
        # Dado que managed=False, podemos seguir usando el ORM para actualizar
        # los registros siempre que la tabla 'sorteo' exista.
        self.stdout.write("Actualizando registros en la base de datos...")
        count = Sorteo.objects.filter(tipo_sorteo="euromilones").update(tipo_sorteo="euromillones")
        self.stdout.write(
            self.style.SUCCESS(
                f"Éxito: Se han actualizado {count} registros de 'euromilones' a 'euromillones'."
            )
        )
