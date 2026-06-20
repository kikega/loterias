from django.core.management.base import BaseCommand
from django.core.management.base import CommandParser
from analytics.ml_services import ejecutar_analisis, tipos_disponibles


class Command(BaseCommand):
    """
    Comando de gestión de Django para entrenar y guardar los modelos LSTM
    de predicción de resultados para los diferentes sorteos de lotería.
    """
    help = "Entrena los modelos LSTM de predicción de loterías."

    def add_arguments(self, parser: CommandParser) -> None:
        """
        Define los argumentos opcionales aceptados por el comando.

        Args:
            parser (CommandParser): Analizador de argumentos del comando de Django.
        """
        parser.add_argument(
            "--tipo",
            type=str,
            help="Tipo de sorteo específico para entrenar (primitiva, gordo, euromillones). Por defecto se entrenan todos.",
        )

    def handle(self, *args, **options) -> None:
        """
        Punto de entrada principal para ejecutar el entrenamiento de los modelos LSTM.

        Args:
            *args: Argumentos posicionales adicionales.
            **options: Opciones y argumentos pasados en la línea de comandos.
        """
        tipo = options.get("tipo")
        if tipo:
            if tipo not in tipos_disponibles():
                self.stdout.write(
                    self.style.ERROR(
                        f"Error: Tipo de sorteo '{tipo}' no disponible. Opciones: {tipos_disponibles()}"
                    )
                )
                return
            tipos = [tipo]
        else:
            tipos = tipos_disponibles()

        for t in tipos:
            self.stdout.write(f"\nEntrenando modelo LSTM para: {t.upper()}...")
            try:
                ejecutar_analisis(t, entrenar=True)
                self.stdout.write(
                    self.style.SUCCESS(f"Modelo para '{t}' entrenado y guardado correctamente.")
                )
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"Error al entrenar el modelo '{t}': {e}"))
