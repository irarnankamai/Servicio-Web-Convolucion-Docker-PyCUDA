
import argparse
import csv
import io
import time
import uuid
import platform
import statistics

from pathlib import Path

import numpy as np
from PIL import Image

from codigo.motor_gpu import MotorGPU


RAIZ = Path(__file__).resolve().parents[1]

RESULTADOS = RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

REPETICIONES = 5
HILOS = 256

RESOLUCION = 4000
FILTRO = "Media"
MASCARA = 9


def ejecutar(motor):

    ruta = (
        RAIZ / "imagenes"
        / f"imagen_{RESOLUCION}.pgm"
    )

    inicio_total = time.perf_counter()

    # Lectura y preparacion de la imagen.
    inicio = time.perf_counter()

    with Image.open(ruta) as imagen:

        entrada = np.asarray(
            imagen.convert("L"),
            dtype=np.float32
        )

    preparacion_ms = (
        time.perf_counter() - inicio
    ) * 1000

    # Procesamiento mediante PyCUDA.
    resultado, tiempos = motor.procesar(
        entrada,
        FILTRO,
        MASCARA,
        HILOS
    )

    # Preparacion del resultado PNG.
    inicio = time.perf_counter()

    salida = np.clip(
        resultado,
        0,
        255
    ).astype(np.uint8)

    memoria = io.BytesIO()

    Image.fromarray(salida).save(
        memoria,
        format="PNG"
    )

    salida_ms = (
        time.perf_counter() - inicio
    ) * 1000

    total_directo_ms = (
        time.perf_counter() - inicio_total
    ) * 1000

    return {
        "preparacion_ms": preparacion_ms,
        "salida_ms": salida_ms,
        "total_directo_ms": total_directo_ms,
        "bytes_resultado": memoria.getbuffer().nbytes,
        **tiempos
    }


def guardar_csv(ruta, registros):

    if not registros:
        return

    campos = list(dict.fromkeys(
        clave
        for registro in registros
        for clave in registro
    ))

    with open(
        ruta,
        "w",
        newline="",
        encoding="utf-8"
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()
        escritor.writerows(registros)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modo",
        required=True,
        choices=["windows", "docker_directo"]
    )

    args = parser.parse_args()

    modo = args.modo

    registros = []

    print("=" * 65)
    print("EXPERIMENTO E1 - EJECUCION DIRECTA")
    print("=" * 65)

    print("Modo:", modo)
    print("Sistema:", platform.platform())
    print("Python:", platform.python_version())
    print("Resolucion:", RESOLUCION)
    print("Filtro:", FILTRO)
    print("Mascara:", MASCARA)
    print("Hilos:", HILOS)

    motor = MotorGPU()

    print("GPU:", motor.nombre_gpu)
    print("Inicializacion ms:", motor.inicializacion_ms)
    print("Compilacion ms:", motor.compilacion_ms)

    try:

        # Primera ejecucion separada.
        primera = ejecutar(motor)

        guardar_csv(
            RESULTADOS / f"arranque_{modo}.csv",
            [{
                "modo": modo,
                "inicializacion_ms": motor.inicializacion_ms,
                "compilacion_ms": motor.compilacion_ms,
                **primera
            }]
        )

        print("\nPrimera ejecucion registrada.")

        # Calentamiento adicional.
        print("Ejecutando calentamiento...")

        ejecutar(motor)

        # Mediciones definitivas.
        for repeticion in range(1, REPETICIONES + 1):

            registro = {
                "request_id": uuid.uuid4().hex,
                "experimento": "E1",
                "modo": modo,
                "filtro": FILTRO,
                "mascara": MASCARA,
                "resolucion": RESOLUCION,
                "hilos": HILOS,
                "repeticion": repeticion,
                "concurrencia": 1,
                "sistema": platform.platform(),
                "python": platform.python_version()
            }

            try:

                tiempos = ejecutar(motor)

                registro.update(tiempos)
                registro["estado"] = "CORRECTO"
                registro["error"] = ""

                print(
                    f"Repeticion {repeticion}: "
                    f"H2D={tiempos['h2d_ms']:.3f} ms | "
                    f"Kernel={tiempos['kernel_ms']:.3f} ms | "
                    f"D2H={tiempos['d2h_ms']:.3f} ms | "
                    f"Total={tiempos['total_directo_ms']:.3f} ms"
                )

            except Exception as error:

                registro["estado"] = "ERROR"
                registro["error"] = str(error)

                print("ERROR:", error)

            registros.append(registro)

    finally:

        motor.cerrar()

        guardar_csv(
            RESULTADOS / f"mediciones_{modo}.csv",
            registros
        )

    correctos = [
        r for r in registros
        if r["estado"] == "CORRECTO"
    ]

    print("\n" + "=" * 65)
    print("RESUMEN")
    print("=" * 65)

    print("Mediciones correctas:", len(correctos))

    if correctos:

        for campo in [
            "h2d_ms",
            "kernel_ms",
            "d2h_ms",
            "gpu_total_ms",
            "total_directo_ms"
        ]:

            valores = [r[campo] for r in correctos]

            mediana = statistics.median(valores)

            desviacion = (
                statistics.stdev(valores)
                if len(valores) > 1
                else float("nan")
            )

            print(
                f"{campo}: "
                f"Mediana={mediana:.3f} ms | "
                f"Desviacion={desviacion:.3f} ms"
            )

    print("\nCSV generado:")
    print(RESULTADOS / f"mediciones_{modo}.csv")


if __name__ == "__main__":
    main()
