from django.contrib.postgres.fields import ArrayField
from django.db import models


class Sorteo(models.Model):
    """
    Modelo de Django que representa un sorteo de lotería registrado.
    La tabla está gestionada externamente (managed=False) en la base de datos PostgreSQL.
    """
    tipo_sorteo = models.CharField(max_length=20, db_index=True)
    fecha = models.DateField(db_index=True)
    bolas = ArrayField(models.SmallIntegerField())
    especiales = ArrayField(models.SmallIntegerField(), null=True, blank=True)

    class Meta:
        managed = False
        db_table = "sorteo"
        unique_together = ("tipo_sorteo", "fecha")
        ordering = ("tipo_sorteo", "fecha")

    def __str__(self) -> str:
        """
        Devuelve la representación en cadena de texto del sorteo.

        Returns:
            str: Identificación formateada del sorteo (tipo, fecha y bolas).
        """
        return f"{self.tipo_sorteo} - {self.fecha} - {self.bolas}"

    def bolas_list(self) -> list[int]:
        """
        Devuelve las bolas principales del sorteo como una lista de enteros estándar.

        Returns:
            list[int]: Lista con las bolas del sorteo.
        """
        return list(self.bolas) if self.bolas else []

    def especiales_list(self) -> list[int]:
        """
        Devuelve las bolas/números especiales del sorteo como una lista de enteros estándar.

        Returns:
            list[int]: Lista con las bolas especiales (o vacía si no hay).
        """
        return list(self.especiales) if self.especiales else []
