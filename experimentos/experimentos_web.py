
import csv
import time
import statistics
import platform
import io

from pathlib import Path

import requests
from PIL import Image


RAIZ = Path(__file__).resolve().parents[1]

RESULTADOS = RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8000/api/procesar"

REPETICIONES = 5
HILOS = 256

CONFIGURACIONES = [
    ("E1", 4000, "Media", 9),

    ("E2", 2000, "Media", 9),
    ("E2", 4000, "Media", 9),
    ("E2", 6000, "Media", 9),

    ("E3", 4000, "Media", 3),
    ("E3", 4000, "Media", 9),
    ("E3", 4000, "Media", 21),
    ("E3", 4000, "Sharpen", 3),
    ("E3", 4000, "Laplaciano", 3)
]

CAMPOS_TIEMPO = [
    "espera_ms",
    "preparacion_ms",
    "reserva_ms",
    "h2d_ms",
    "kernel_ms",
    "d2h_ms",
    "gpu_total_ms",
    "motor_total_ms",
    "salida_ms",
    "servidor_ms",
    "solicitud_ms",
    "arranque_ms"
]


def procesar(resolucion, filtro, mascara):

    ruta = (
        RAIZ / "imagenes"
        / f"imagen_{resolucion}.pgm"
    )

    with open(ruta, "rb") as archivo:

        inicio = time.perf_counter()

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
                "salida": "png"
            },
            timeout=300
        )

        cliente_ms = (
            time.perf_counter() - inicio
        ) * 1000

    datos = {
        "request_id": respuesta.headers.get(
            "X-Request-ID", ""
        ),
        "http": respuesta.status_code,
        "cliente_ms": cliente_ms,
        "bytes_resultado": len(respuesta.content),
        "estado": (
            "CORRECTO"
            if respuesta.status_code == 200
            else "ERROR"
        ),
        "error": ""
    }

    if respuesta.status_code != 200:

        datos["error"] = respuesta.text[:500]

        return datos, respuesta.content

    for campo in CAMPOS_TIEMPO:

        encabezado = "X-" + campo.replace("_", "-")

        valor = respuesta.headers.get(encabezado)

        datos[campo] = (
            float(valor)
            if valor is not None
            else ""
        )

    for campo in [
        "ancho",
        "alto",
        "hilos",
        "bloques",
        "bloques_x",
        "bloques_y"
    ]:

        valor = respuesta.headers.get(
            "X-" + campo.replace("_", "-")
        )

        datos[campo] = (
            int(valor)
            if valor is not None
            else ""
        )

    datos["gpu"] = respuesta.headers.get(
        "X-GPU", ""
    )

    datos["arranque_frio"] = respuesta.headers.get(
        "X-Arranque-Frio", ""
    )

    # Comprobacion del formato de salida.
    try:

        with Image.open(
            io.BytesIO(respuesta.content)
        ) as imagen:

            imagen.verify()

        with Image.open(
            io.BytesIO(respuesta.content)
        ) as imagen:

            if imagen.size != (resolucion, resolucion):

                raise ValueError(
                    "Dimensiones de salida incorrectas"
                )

            if imagen.format != "PNG":

                raise ValueError(
                    "Formato de salida incorrecto"
                )

        if not datos["request_id"]:

            raise ValueError(
                "Identificador de solicitud ausente"
            )

        if any(
            datos[campo] == ""
            for campo in CAMPOS_TIEMPO
        ):

            raise ValueError(
                "Faltan encabezados de medicion"
            )

        if datos["hilos"] != HILOS:

            raise ValueError(
                "Cantidad de hilos incorrecta"
            )

    except Exception as error:

        datos["estado"] = "ERROR"
        datos["error"] = str(error)

    return datos, respuesta.content


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

    registros = []
    resumen = []

    print("=" * 70)
    print("EXPERIMENTOS DE RENDIMIENTO - SERVICIO WEB")
    print("=" * 70)

    try:

        for experimento, resolucion, filtro, mascara in CONFIGURACIONES:

            print("\n" + "=" * 70)

            print(
                f"{experimento} | "
                f"{resolucion}x{resolucion} | "
                f"{filtro} {mascara}x{mascara}"
            )

            print("=" * 70)

            # Calentamiento por configuracion.
            print("Calentamiento...")

            calentamiento, _ = procesar(
                resolucion,
                filtro,
                mascara
            )

            if calentamiento["estado"] != "CORRECTO":

                registros.append({
                    "experimento": experimento,
                    "modo": "servicio_web",
                    "resolucion": resolucion,
                    "filtro": filtro,
                    "mascara": mascara,
                    "repeticion": 0,
                    "concurrencia": 1,
                    **calentamiento
                })

                print(
                    "Calentamiento fallido:",
                    calentamiento["error"]
                )

                continue

            grupo = []

            for repeticion in range(
                1, REPETICIONES + 1
            ):

                registro = {
                    "experimento": experimento,
                    "modo": "servicio_web",
                    "resolucion": resolucion,
                    "filtro": filtro,
                    "mascara": mascara,
                    "hilos": HILOS,
                    "repeticion": repeticion,
                    "concurrencia": 1,
                    "cliente": "Docker localhost",
                    "sistema_cliente": platform.platform()
                }

                try:

                    datos, contenido = procesar(
                        resolucion,
                        filtro,
                        mascara
                    )

                    registro.update(datos)

                    if datos["estado"] == "CORRECTO":

                        print(
                            f"Rep {repeticion}: "
                            f"Kernel={datos['kernel_ms']:.3f} ms | "
                            f"Servidor={datos['servidor_ms']:.3f} ms | "
                            f"Cliente={datos['cliente_ms']:.3f} ms"
                        )

                        if repeticion == 1 and experimento == "E3":

                            nombre = (
                                f"{filtro}_{mascara}x{mascara}"
                                f"_{resolucion}.png"
                            )

                            (RESULTADOS / nombre).write_bytes(
                                contenido
                            )

                    else:

                        print("ERROR:", datos["error"])

                except Exception as error:

                    registro["estado"] = "ERROR"
                    registro["error"] = str(error)

                    print("ERROR:", error)

                registros.append(registro)
                grupo.append(registro)

            correctos = [
                r for r in grupo
                if r["estado"] == "CORRECTO"
            ]

            fila = {
                "experimento": experimento,
                "modo": "servicio_web",
                "resolucion": resolucion,
                "filtro": filtro,
                "mascara": mascara,
                "hilos": HILOS,
                "correctas": len(correctos),
                "errores": REPETICIONES - len(correctos)
            }

            for campo in CAMPOS_TIEMPO + ["cliente_ms"]:

                valores = [
                    r[campo]
                    for r in correctos
                    if isinstance(r.get(campo), (int, float))
                ]

                if valores:

                    fila[campo + "_mediana"] = (
                        statistics.median(valores)
                    )

                    fila[campo + "_desviacion"] = (
                        statistics.stdev(valores)
                        if len(valores) > 1
                        else ""
                    )

            resumen.append(fila)

    finally:

        guardar_csv(
            RESULTADOS / "mediciones_web.csv",
            registros
        )

        guardar_csv(
            RESULTADOS / "resumen_web.csv",
            resumen
        )

    correctas = sum(
        r["estado"] == "CORRECTO"
        for r in registros
    )

    print("\n" + "=" * 70)
    print("RESUMEN GENERAL")
    print("=" * 70)

    print("Mediciones previstas:", 45)
    print("Mediciones registradas:", len(registros))
    print("Correctas:", correctas)
    print("Errores:", len(registros) - correctas)

    if correctas == 45:

        print("EXPERIMENTOS COMPLETADOS CORRECTAMENTE")

    else:

        print("EXPERIMENTOS INCOMPLETOS O CON ERRORES")

    print("\nArchivos generados:")
    print(RESULTADOS / "mediciones_web.csv")
    print(RESULTADOS / "resumen_web.csv")


if __name__ == "__main__":
    main()
