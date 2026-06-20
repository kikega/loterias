from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    """
    Comando de gestión de Django para inicializar un usuario administrador por defecto
    ('admin' / 'admin1234') en entornos locales de desarrollo si no existe previamente.
    """
    help = "Crea un usuario administrador por defecto si no existe."

    def handle(self, *args, **options) -> None:
        """
        Crea el superusuario si no se encuentra registrado en el modelo de usuario.

        Args:
            *args: Argumentos posicionales adicionales.
            **options: Opciones pasadas al comando.
        """
        User = get_user_model()
        username = "admin"
        email = "admin@example.com"
        password = "admin1234"

        if not User.objects.filter(username=username).exists():
            self.stdout.write(f"Creando usuario administrador '{username}'...")
            User.objects.create_superuser(
                username=username, email=email, password=password
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Éxito: Se ha creado el usuario '{username}' con contraseña '{password}'."
                )
            )
        else:
            self.stdout.write(
                self.style.WARNING(
                    f"El usuario '{username}' ya existe. No se realizaron cambios."
                )
            )
