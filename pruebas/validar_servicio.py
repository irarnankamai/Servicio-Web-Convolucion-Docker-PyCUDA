
import csv
import io
import sys
import time
import hashlib

from pathlib import Path

import numpy as np
import requests

from PIL import Image


RAIZ = Path(__file__).resolve().parents[1]

RESULTADOS = RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8000/api/procesar"

IMAGEN = RAIZ / "imagenes" / "imagen_2000.pgm"

registros = []


def ejecutar(
    nombre,
    contenido,
    esperado,
    filtro="Media",
    mascara=3,
    archivo="imagen.pgm",
    salida="png"
):

    datos = {
        "filtro": filtro,
        "mascara": str(mascara),
        "salida": salida
    }

    archivos = None

    if contenido is not None:

        archivos = {
            "imagen": (
                archivo,
                contenido,
                "application/octet-stream"
            )
        }

    inicio = time.perf_counter()

    respuesta = requests.post(
        URL,
        files=archivos,
        data=datos,
        timeout=300
    )

    tiempo_ms = (
        time.perf_counter() - inicio
    ) * 1000

    valido = respuesta.status_code == esperado

    request_id = respuesta.headers.get(
        "X-Request-ID", ""
    )

    mensaje = ""

    if respuesta.status_code != 200:
        mensaje = respuesta.text[:500]

    registro = {
        "prueba": nombre,
        "filtro": filtro,
        "mascara": mascara,
        "http_esperado": esperado,
        "http_obtenido": respuesta.status_code,
        "tiempo_cliente_ms": tiempo_ms,
        "request_id": request_id,
        "bytes_respuesta": len(respuesta.content),
        "validacion": "CORRECTA" if valido else "ERROR",
        "mensaje": mensaje
    }

    registros.append(registro)

    print("-" * 65)
    print("Prueba:", nombre)
    print("HTTP esperado:", esperado)
    print("HTTP obtenido:", respuesta.status_code)
    print("Tiempo cliente ms:", round(tiempo_ms, 3))
    print("Request ID:", request_id)
    print("Validacion:", registro["validacion"])

    if mensaje:
        print("Respuesta:", mensaje)

    return respuesta


def guardar_csv():

    ruta = RESULTADOS / "validacion_servicio.csv"

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

    print("\nCSV generado:", ruta)


def crear_imagen_excedida():

    memoria = io.BytesIO()

    imagen = Image.new(
        "L",
        (6001, 100),
        color=128
    )

    imagen.save(memoria, format="PNG")

    return memoria.getvalue()


def validar_aislamiento(contenido):

    primera = ejecutar(
        "Aislamiento solicitud A",
        contenido,
        200,
        filtro="Media",
        mascara=3,
        salida="npy"
    )

    segunda = ejecutar(
        "Aislamiento solicitud B",
        contenido,
        200,
        filtro="Laplaciano",
        mascara=3,
        salida="npy"
    )

    id_a = primera.headers.get("X-Request-ID", "")
    id_b = segunda.headers.get("X-Request-ID", "")

    ids_distintos = bool(
        id_a and id_b and id_a != id_b
    )

    hash_a = hashlib.sha256(
        primera.content
    ).hexdigest()

    hash_b = hashlib.sha256(
        segunda.content
    ).hexdigest()

    resultados_distintos = hash_a != hash_b

    formato_correcto = False

    if primera.status_code == segunda.status_code == 200:

        matriz_a = np.load(
            io.BytesIO(primera.content),
            allow_pickle=False
        )

        matriz_b = np.load(
            io.BytesIO(segunda.content),
            allow_pickle=False
        )

        formato_correcto = (
            matriz_a.shape == (2000, 2000)
            and matriz_b.shape == (2000, 2000)
            and matriz_a.dtype == np.float32
            and matriz_b.dtype == np.float32
        )

    correcto = (
        ids_distintos
        and resultados_distintos
        and formato_correcto
    )

    print("\n" + "=" * 65)
    print("COMPROBACION DE AISLAMIENTO")
    print("ID A:", id_a)
    print("ID B:", id_b)
    print("Identificadores distintos:", ids_distintos)
    print("SHA256 A:", hash_a)
    print("SHA256 B:", hash_b)
    print("Resultados distintos:", resultados_distintos)
    print("Formato correcto:", formato_correcto)
    print("Aislamiento:", "CORRECTO" if correcto else "ERROR")

    if not correcto:
        registros[-1]["validacion"] = "ERROR"

    return correcto


def main():

    print("=" * 65)
    print("VALIDACION DEL SERVICIO WEB - PRACTICA 03")
    print("=" * 65)

    contenido = IMAGEN.read_bytes()

    try:

        ejecutar(
            "Imagen valida",
            contenido,
            200
        )

        ejecutar(
            "Archivo invalido",
            b"Esto no es una imagen",
            400,
            archivo="archivo.txt"
        )

        ejecutar(
            "Archivo vacio",
            b"",
            400
        )

        ejecutar(
            "Solicitud sin imagen",
            None,
            400
        )

        ejecutar(
            "Filtro desconocido",
            contenido,
            400,
            filtro="Desconocido"
        )

        ejecutar(
            "Mascara Media invalida",
            contenido,
            400,
            filtro="Media",
            mascara=5
        )

        ejecutar(
            "Mascara Sharpen invalida",
            contenido,
            400,
            filtro="Sharpen",
            mascara=9
        )

        ejecutar(
            "Mascara Laplaciano invalida",
            contenido,
            400,
            filtro="Laplaciano",
            mascara=21
        )

        ejecutar(
            "Archivo superior a 40 MiB",
            b"x" * (40 * 1024 * 1024 + 1),
            413,
            archivo="archivo_grande.pgm"
        )

        ejecutar(
            "Dimensiones superiores a 6000",
            crear_imagen_excedida(),
            400,
            archivo="imagen_excedida.png"
        )

        recuperacion = ejecutar(
            "Recuperacion del servicio",
            contenido,
            200,
            filtro="Media",
            mascara=9
        )

        if recuperacion.status_code == 200:

            ruta = RESULTADOS / "recuperacion_servicio.png"

            ruta.write_bytes(recuperacion.content)

        validar_aislamiento(contenido)

    finally:

        guardar_csv()

    correctas = sum(
        r["validacion"] == "CORRECTA"
        for r in registros
    )

    errores = len(registros) - correctas

    print("\n" + "=" * 65)
    print("RESUMEN FINAL")
    print("=" * 65)
    print("Solicitudes realizadas:", len(registros))
    print("Correctas:", correctas)
    print("Errores:", errores)

    if errores == 0 and len(registros) == 13:

        print("VALIDACION GENERAL: CORRECTA")
        return 0

    print("VALIDACION GENERAL: ERROR O INCOMPLETA")
    return 1


if __name__ == "__main__":
    sys.exit(main())
