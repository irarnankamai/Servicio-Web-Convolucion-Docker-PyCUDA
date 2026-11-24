
import csv
import io
import sys
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from codigo.motor_gpu import MotorGPU
from codigo.filtros_python import crear_mascara


# Configuracion general.
RAIZ = Path(__file__).resolve().parents[1]

RESULTADOS = RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8000/api/procesar"

TOLERANCIA = 1e-4
HILOS = 256

CONFIGURACIONES = [
    ("Media", 3),
    ("Media", 9),
    ("Media", 21),
    ("Sharpen", 3),
    ("Laplaciano", 3)
]

registros = []


def convolucion_referencia(entrada, mascara):
    """Convolucion secuencial independiente en CPU."""

    entrada = np.asarray(entrada, dtype=np.float32)
    mascara = np.asarray(mascara, dtype=np.float32)

    alto, ancho = entrada.shape

    k = mascara.shape[0]
    radio = k // 2

    salida = np.zeros_like(entrada)

    for y in range(alto):
        for x in range(ancho):

            suma = np.float32(0)

            for j in range(k):
                for i in range(k):

                    yy = y + j - radio
                    xx = x + i - radio

                    if 0 <= yy < alto and 0 <= xx < ancho:

                        producto = np.float32(
                            entrada[yy, xx] * mascara[j, i]
                        )

                        suma = np.float32(suma + producto)

            salida[y, x] = suma

    return salida


def comparar(
    prueba,
    filtro,
    mascara,
    referencia,
    resultado,
    request_id=""
):

    if referencia.shape != resultado.shape:
        raise ValueError("Las dimensiones no coinciden")

    if not np.all(np.isfinite(resultado)):
        raise ValueError("Resultado con valores no finitos")

    diferencia = np.abs(
        referencia.astype(np.float64)
        - resultado.astype(np.float64)
    )

    error_maximo = float(np.max(diferencia))
    error_medio = float(np.mean(diferencia))

    valido = error_maximo <= TOLERANCIA

    registro = {
        "prueba": prueba,
        "filtro": filtro,
        "mascara": mascara,
        "ancho": referencia.shape[1],
        "alto": referencia.shape[0],
        "hilos": HILOS,
        "error_maximo": error_maximo,
        "error_medio": error_medio,
        "tolerancia": TOLERANCIA,
        "validacion": "CORRECTA" if valido else "ERROR",
        "request_id": request_id
    }

    registros.append(registro)

    print("-" * 65)
    print("Prueba:", prueba)
    print("Filtro:", filtro)
    print("Mascara:", mascara)
    print("Resolucion:", referencia.shape)
    print("Error maximo:", f"{error_maximo:.10e}")
    print("Error medio:", f"{error_medio:.10e}")
    print("Tolerancia:", TOLERANCIA)
    print("Validacion:", registro["validacion"])


def procesar_servicio(ruta, filtro, mascara):

    with open(ruta, "rb") as archivo:

        respuesta = requests.post(
            URL,
            files={
                "imagen": (
                    ruta.name,
                    archivo,
                    "application/octet-stream"
                )
            },
            data={
                "filtro": filtro,
                "mascara": str(mascara),
                "salida": "npy"
            },
            timeout=300
        )

    if respuesta.status_code != 200:
        raise RuntimeError(
            f"HTTP {respuesta.status_code}: "
            f"{respuesta.text[:500]}"
        )

    if not respuesta.content.startswith(b"\x93NUMPY"):
        raise RuntimeError(
            "El servicio no devolvio un archivo NPY. "
            "Revisar el parametro salida de la API."
        )

    resultado = np.load(
        io.BytesIO(respuesta.content),
        allow_pickle=False
    )

    if resultado.dtype != np.float32:
        raise ValueError("El servicio no devolvio Float32")

    request_id = respuesta.headers.get("X-Request-ID", "")

    return resultado, request_id


def guardar_csv():

    ruta = RESULTADOS / "validacion_numerica.csv"

    if registros:

        with open(
            ruta,
            "w",
            newline="",
            encoding="utf-8"
        ) as archivo:

            escritor = csv.DictWriter(
                archivo,
                fieldnames=list(registros[0].keys())
            )

            escritor.writeheader()
            escritor.writerows(registros)

    print("\nArchivo CSV:", ruta)


def main():

    print("=" * 65)
    print("VALIDACION NUMERICA - PRACTICA 03")
    print("=" * 65)

    motor = MotorGPU()

    print("GPU:", motor.nombre_gpu)
    print("Inicializacion ms:", motor.inicializacion_ms)
    print("Compilacion ms:", motor.compilacion_ms)

    try:

        # PRUEBA 1: Matriz pequena.

        matriz = np.ones((5, 5), dtype=np.float32)

        esperado = np.array([
            [4, 6, 6, 6, 4],
            [6, 9, 9, 9, 6],
            [6, 9, 9, 9, 6],
            [6, 9, 9, 9, 6],
            [4, 6, 6, 6, 4]
        ], dtype=np.float32) / np.float32(9)

        resultado, _ = motor.procesar(
            matriz, "Media", 3, HILOS
        )

        comparar(
            "Matriz manual 5x5",
            "Media",
            3,
            esperado,
            resultado
        )

        # PRUEBA 2: CPU frente a GPU.
        # Matriz pequena y matriz irregular.

        rng = np.random.default_rng(42)

        irregular = rng.integers(
            0, 256,
            size=(37, 53)
        ).astype(np.float32)

        matrices = [
            ("CPU vs GPU 5x5", matriz),
            ("CPU vs GPU 37x53", irregular)
        ]

        for nombre, entrada in matrices:

            for filtro, k in CONFIGURACIONES:

                mascara = crear_mascara(filtro, k)

                referencia = convolucion_referencia(
                    entrada,
                    mascara
                )

                resultado, _ = motor.procesar(
                    entrada,
                    filtro,
                    k,
                    HILOS
                )

                comparar(
                    nombre,
                    filtro,
                    k,
                    referencia,
                    resultado
                )

        # PRUEBA 3: PyCUDA directo frente al servicio HTTP.

        ruta = RAIZ / "imagenes" / "imagen_2000.pgm"

        with Image.open(ruta) as imagen:

            entrada = np.asarray(
                imagen.convert("L"),
                dtype=np.float32
            )

        for filtro, k in CONFIGURACIONES:

            directo, _ = motor.procesar(
                entrada,
                filtro,
                k,
                HILOS
            )

            servicio, request_id = procesar_servicio(
                ruta,
                filtro,
                k
            )

            comparar(
                "Directo vs Servicio HTTP",
                filtro,
                k,
                directo,
                servicio,
                request_id
            )

    finally:

        motor.cerrar()
        guardar_csv()

    correctas = sum(
        r["validacion"] == "CORRECTA"
        for r in registros
    )

    errores = len(registros) - correctas

    print("\n" + "=" * 65)
    print("RESUMEN FINAL")
    print("=" * 65)
    print("Total de validaciones:", len(registros))
    print("Correctas:", correctas)
    print("Errores:", errores)

    if errores == 0 and len(registros) == 16:
        print("VALIDACION GENERAL: CORRECTA")
        return 0

    print("VALIDACION GENERAL: ERROR O INCOMPLETA")
    return 1


if __name__ == "__main__":
    sys.exit(main())
