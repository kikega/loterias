import json


def procesar_loteria(data):
    print(f"Sorteo: {data['sorteo']}")
    print(f"Archivo: {data['fichero']}")
    print(f"Columnas: {data['columnas']}")
    print(f"Números: {data['numeros']}")
    print(f"Números especiales: {data['numeros_especiales']}")
    print("-"*25)

def main():
    try:
        with open('sorteos.json', "r", encoding="utf-8") as archivo:
            data = json.load(archivo)
    except Exception as e:
        print(f"Error: No se pudo abrir el archivo 'sorteos.json' por {e}")

    for sorteo in data:
        print(sorteo)
        procesar_loteria(sorteo)

if __name__ == "__main__":
    main()