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


class PrediccionSemanal(models.Model):
    """
    Modelo que representa el conjunto de predicciones para una semana ISO
    específica y un tipo de sorteo determinado.
    """
    tipo_sorteo = models.CharField(max_length=20, db_index=True)
    anio = models.IntegerField(db_index=True)
    semana = models.IntegerField(db_index=True)
    fecha_creacion = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "prediccion_semanal"
        unique_together = ("tipo_sorteo", "anio", "semana")
        ordering = ("-anio", "-semana", "tipo_sorteo")

    def __str__(self) -> str:
        return f"Predicción {self.tipo_sorteo} - {self.anio}-W{self.semana:02d}"


class CombinacionPredicha(models.Model):
    """
    Modelo que representa una de las 3 combinaciones estimadas sugeridas
    para un sorteo semanal, registrando los aciertos de los sorteos reales.
    """
    prediccion_semanal = models.ForeignKey(
        PrediccionSemanal, on_delete=models.CASCADE, related_name="combinaciones"
    )
    orden = models.IntegerField()  # 1, 2 o 3
    estrategia = models.CharField(max_length=50)  # 'lstm_pura', 'tendencia_pura', 'hibrida_adaptativa'
    bolas = ArrayField(models.SmallIntegerField())
    especiales = ArrayField(models.SmallIntegerField(), null=True, blank=True)
    
    # Detalle de aciertos en formato JSON por fecha de sorteo
    # Estructura: {"AAAA-MM-DD": {"bolas": [1,2], "especiales": [3], "total_bolas": 2, "total_especiales": 1}}
    aciertos_por_sorteo = models.JSONField(default=dict, blank=True)
    procesado = models.BooleanField(default=False)

    class Meta:
        db_table = "combinacion_predicha"
        ordering = ("prediccion_semanal", "orden")

    def __str__(self) -> str:
        return f"Comb {self.orden} ({self.estrategia}) - {self.prediccion_semanal}"


from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=Sorteo)
def al_guardar_sorteo(sender, instance, created, **kwargs):
    """
    Señal que se dispara después de guardar un sorteo.
    Calcula los aciertos de la predicción de la semana correspondiente
    e inicia el reentrenamiento de la red LSTM de manera asíncrona.
    """
    if created:
        from .ml_services import evaluar_predicciones_semana, entrenar_modelo_asincrono
        # Evaluar predicción de la semana
        evaluar_predicciones_semana(instance)
        # Reentrenar modelo de manera asíncrona
        entrenar_modelo_asincrono(instance.tipo_sorteo)


