"""
Estudio sobre probabilidad
"""
import pandas as pd

iterador = iter([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print("Primer bucle")
for i in iterador:
    print(i)
    if i == 8:
        break

print("Segundo bucle")
for i in iterador:
    print(i)
