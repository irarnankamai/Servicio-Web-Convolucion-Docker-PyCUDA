
import csv
import io
import time
import hashlib
import threading
import statistics

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
import numpy as np


RAIZ = Path(__file__).resolve().parents[1]

RESULTADOS = RAIZ / "resultados"
RESULTADOS.mkdir(parents=True, exist_ok=True)

URL = "http://localhost:8000/api/procesar"

RONDAS = 5

registros = []
registros_rondas = []
registros_adicionales = []

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


def solicitud(
    cliente,
    resolucion=4000,
    filtro="Media",
    mascara=9,
    salida="png",
    barrera=None
):

    ruta = (
        RAIZ / "imagenes"
        / f"imagen_{resolucion}.pgm"
    )

    registro = {
        "cliente": cliente,
        "resolucion": resolucion,
        "filtro": filtro,
        "mascara": mascara,
        "hilos": 256,
        "formato_salida": salida,
        "ubicacion_cliente": "Docker localhost",
        "estado": "ERROR",
        "error": ""
    }

    contenido = b""

    with open(ruta, "rb") as archivo:

        if barrera is not None:
            barrera.wait(timeout=30)

        inicio = time.perf_counter()

        try:

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
                    "salida": salida
                },
                timeout=300
            )

            registro["cliente_ms"] = (
                time.perf_counter() - inicio
            ) * 1000

            contenido = respuesta.content

            registro["http"] = respuesta.status_code

            registro["request_id"] = (
                respuesta.headers.get(
                    "X-Request-ID", ""
                )
            )

            registro["bytes_resultado"] = len(
                contenido
            )

            if respuesta.status_code == 200:

                registro["estado"] = "CORRECTO"

                registro["sha256"] = (
                    hashlib.sha256(contenido)
                    .hexdigest()
                )

                for campo in CAMPOS_TIEMPO:

                    encabezado = (
                        "X-" + campo.replace("_", "-")
                    )

                    valor = respuesta.headers.get(
                        encabezado
                    )

                    registro[campo] = (
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

                    registro[campo] = (
                        int(valor)
                        if valor is not None
                        else ""
                    )

                registro["gpu"] = (
                    respuesta.headers.get("X-GPU", "")
                )

                if not registro["request_id"]:

                    registro["estado"] = "ERROR"
                    registro["error"] = "ID ausente"

                elif any(
                    registro[campo] == ""
                    for campo in CAMPOS_TIEMPO
                ):

                    registro["estado"] = "ERROR"
                    registro["error"] = "Faltan tiempos"

                elif (
                    salida == "png"
                    and not contenido.startswith(
                        b"\x89PNG\r\n\x1a\n"
                    )
                ):

                    registro["estado"] = "ERROR"
                    registro["error"] = "PNG invalido"

            else:

                registro["error"] = respuesta.text[:500]

                try:

                    registro["request_id"] = (
                        respuesta.json().get("id", "")
                    )

                except Exception:
                    pass

                if respuesta.status_code == 429:
                    registro["estado"] = "RECHAZADO"

        except Exception as error:

            registro["cliente_ms"] = (
                time.perf_counter() - inicio
            ) * 1000

            registro["http"] = 0
            registro["error"] = str(error)

    return registro, contenido


def ejecutar_ronda(clientes, configuraciones=None):

    barrera = threading.Barrier(clientes + 1)

    if configuraciones is None:

        configuraciones = [
            (4000, "Media", 9, "png")
            for _ in range(clientes)
        ]

    with ThreadPoolExecutor(
        max_workers=clientes
    ) as ejecutor:

        futuros = []

        for i, config in enumerate(configuraciones):

            resolucion, filtro, mascara, salida = config

            futuro = ejecutor.submit(
                solicitud,
                i + 1,
                resolucion,
                filtro,
                mascara,
                salida,
                barrera
            )

            futuros.append(futuro)

        inicio = time.perf_counter()

        barrera.wait(timeout=30)

        respuestas = [
            futuro.result()
            for futuro in futuros
        ]

        duracion_s = (
            time.perf_counter() - inicio
        )

    return respuestas, duracion_s


def guardar_csv(nombre, filas):

    if not filas:
        return

    campos = list(dict.fromkeys(
        clave
        for fila in filas
        for clave in fila
    ))

    ruta = RESULTADOS / nombre

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
        escritor.writerows(filas)

    print("CSV generado:", ruta)


def experimento_concurrencia():

    print("\nEXPERIMENTO E4 - CONCURRENCIA")

    for clientes in [1, 2]:

        print("\n" + "=" * 65)
        print("CLIENTES SIMULTANEOS:", clientes)
        print("=" * 65)

        print("Calentamiento...")

        calentamiento, _ = ejecutar_ronda(clientes)

        if any(
            r["estado"] != "CORRECTO"
            for r, _ in calentamiento
        ):

            print("ADVERTENCIA: calentamiento fallido")

        for ronda in range(1, RONDAS + 1):

            respuestas, duracion_s = ejecutar_ronda(
                clientes
            )

            exitosas = sum(
                r["estado"] == "CORRECTO"
                for r, _ in respuestas
            )

            throughput = exitosas / duracion_s

            ids = [
                r.get("request_id", "")
                for r, _ in respuestas
                if r["estado"] == "CORRECTO"
            ]

            ids_unicos = (
                all(ids)
                and len(ids) == len(set(ids))
            )

            hashes = [
                r.get("sha256", "")
                for r, _ in respuestas
                if r["estado"] == "CORRECTO"
            ]

            salidas_iguales = (
                exitosas == clientes
                and len(set(hashes)) == 1
            )

            for registro, _ in respuestas:

                registro.update({
                    "experimento": "E4",
                    "modo": "servicio_web",
                    "ronda": ronda,
                    "repeticion": ronda,
                    "concurrencia": clientes,
                    "duracion_ronda_s": duracion_s,
                    "throughput_req_s": throughput
                })

                registros.append(registro)

                print(
                    f"Ronda {ronda} | "
                    f"Cliente {registro['cliente']} | "
                    f"HTTP {registro.get('http', 0)} | "
                    f"Latencia "
                    f"{registro['cliente_ms']:.3f} ms | "
                    f"Espera "
                    f"{registro.get('espera_ms', '')} ms"
                )

            registros_rondas.append({
                "experimento": "E4",
                "concurrencia": clientes,
                "ronda": ronda,
                "solicitudes": clientes,
                "exitosas": exitosas,
                "fallidas": clientes - exitosas,
                "duracion_ronda_s": duracion_s,
                "throughput_req_s": throughput,
                "ids_unicos": ids_unicos,
                "salidas_iguales": salidas_iguales
            })

            print(
                f"Duracion ronda: {duracion_s:.3f} s | "
                f"Throughput: {throughput:.3f} req/s"
            )


def validar_aislamiento():

    print("\n" + "=" * 65)
    print("AISLAMIENTO CONCURRENTE")
    print("=" * 65)

    configuraciones = [
        (2000, "Media", 3, "npy"),
        (2000, "Laplaciano", 3, "npy")
    ]

    # Referencias individuales.
    referencias = []

    for config in configuraciones:

        registro, contenido = solicitud(
            1, *config
        )

        if registro["estado"] != "CORRECTO":

            raise RuntimeError(
                "No se obtuvo referencia individual"
            )

        referencias.append(
            np.load(
                io.BytesIO(contenido),
                allow_pickle=False
            )
        )

    # Solicitudes simultaneas con filtros diferentes.
    respuestas, _ = ejecutar_ronda(
        2,
        configuraciones
    )

    ids = [
        registro.get("request_id", "")
        for registro, _ in respuestas
    ]

    ids_unicos = (
        all(ids)
        and len(ids) == len(set(ids))
    )

    for i, (registro, contenido) in enumerate(respuestas):

        correcto = False

        error_maximo = ""
        error_medio = ""

        if registro["estado"] == "CORRECTO":

            resultado = np.load(
                io.BytesIO(contenido),
                allow_pickle=False
            )

            referencia = referencias[i]

            if (
                resultado.shape == referencia.shape
                and resultado.dtype == np.float32
                and np.all(np.isfinite(resultado))
            ):

                diferencia = np.abs(
                    resultado.astype(np.float64)
                    - referencia.astype(np.float64)
                )

                error_maximo = float(
                    np.max(diferencia)
                )

                error_medio = float(
                    np.mean(diferencia)
                )

                correcto = (
                    error_maximo <= 1e-4
                    and ids_unicos
                )

        registro.update({
            "prueba": "Aislamiento concurrente",
            "concurrencia": 2,
            "error_maximo": error_maximo,
            "error_medio": error_medio,
            "ids_unicos": ids_unicos,
            "validacion": (
                "CORRECTA" if correcto else "ERROR"
            )
        })

        registros_adicionales.append(registro)

        print("Filtro:", registro["filtro"])
        print("Request ID:", registro.get("request_id"))
        print("Error maximo:", error_maximo)
        print("Error medio:", error_medio)
        print("Validacion:", registro["validacion"])


def validar_sobrecarga():

    print("\n" + "=" * 65)
    print("PRUEBA DE SOBRECARGA")
    print("=" * 65)

    configuraciones = [
        (6000, "Media", 21, "png")
        for _ in range(4)
    ]

    total_rechazadas = 0

    # Hasta tres intentos para observar el limite.
    for intento in range(1, 4):

        respuestas, _ = ejecutar_ronda(
            4,
            configuraciones
        )

        rechazadas = 0

        for registro, _ in respuestas:

            registro.update({
                "prueba": "Sobrecarga",
                "intento": intento,
                "concurrencia": 4
            })

            registros_adicionales.append(registro)

            if registro.get("http") == 429:
                rechazadas += 1

            print(
                f"Intento {intento} | "
                f"Cliente {registro['cliente']} | "
                f"HTTP {registro.get('http', 0)} | "
                f"{registro['estado']}"
            )

        total_rechazadas += rechazadas

        if rechazadas > 0:
            break

    print("Rechazos HTTP 429:", total_rechazadas)

    if total_rechazadas == 0:

        print(
            "No se observo saturacion. "
            "No afirmar que el rechazo fue validado."
        )

    # Comprobar recuperacion despues de sobrecarga.
    registro, contenido = solicitud(
        1, 2000, "Media", 9, "png"
    )

    registro["prueba"] = "Recuperacion tras sobrecarga"
    registro["concurrencia"] = 1

    registros_adicionales.append(registro)

    print(
        "Recuperacion HTTP:",
        registro.get("http", 0)
    )

    if registro["estado"] == "CORRECTO":

        (
            RESULTADOS
            / "recuperacion_concurrencia.png"
        ).write_bytes(contenido)


def generar_resumen():

    resumen = []

    for clientes in [1, 2]:

        grupo = [
            r for r in registros
            if r["concurrencia"] == clientes
            and r["estado"] == "CORRECTO"
        ]

        rondas = [
            r for r in registros_rondas
            if r["concurrencia"] == clientes
        ]

        fila = {
            "concurrencia": clientes,
            "rondas": len(rondas),
            "solicitudes_previstas": clientes * RONDAS,
            "solicitudes_correctas": len(grupo),
            "solicitudes_fallidas": (
                clientes * RONDAS - len(grupo)
            )
        }

        for campo in [
            "cliente_ms",
            "espera_ms",
            "kernel_ms",
            "servidor_ms",
            "gpu_total_ms"
        ]:

            valores = [
                r[campo]
                for r in grupo
                if isinstance(
                    r.get(campo), (int, float)
                )
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

        tasas = [
            r["throughput_req_s"]
            for r in rondas
        ]

        if tasas:

            fila["throughput_mediana"] = (
                statistics.median(tasas)
            )

            fila["throughput_desviacion"] = (
                statistics.stdev(tasas)
                if len(tasas) > 1
                else ""
            )

        resumen.append(fila)

        print("\nCLIENTES:", clientes)

        for clave, valor in fila.items():

            if isinstance(valor, float):

                print(f"{clave}: {valor:.3f}")

            else:

                print(f"{clave}: {valor}")

    guardar_csv(
        "resumen_concurrencia.csv",
        resumen
    )


def main():

    print("=" * 65)
    print("EXPERIMENTOS DE CONCURRENCIA - PRACTICA 03")
    print("=" * 65)

    try:

        experimento_concurrencia()

        validar_aislamiento()

        validar_sobrecarga()

    finally:

        guardar_csv(
            "concurrencia.csv",
            registros
        )

        guardar_csv(
            "rondas_concurrencia.csv",
            registros_rondas
        )

        guardar_csv(
            "validacion_concurrencia.csv",
            registros_adicionales
        )

        generar_resumen()

    correctas = sum(
        r["estado"] == "CORRECTO"
        for r in registros
    )

    print("\n" + "=" * 65)
    print("RESUMEN FINAL")
    print("=" * 65)
    print("Mediciones E4 previstas: 15")
    print("Mediciones E4 registradas:", len(registros))
    print("Mediciones E4 correctas:", correctas)
    print("Mediciones E4 fallidas:", len(registros) - correctas)


if __name__ == "__main__":
    main()
