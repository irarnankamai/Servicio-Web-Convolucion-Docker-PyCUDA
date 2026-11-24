
import atexit
import io
import logging
import threading
import time
import uuid
import zipfile

import numpy as np

from flask import (
    Flask,
    jsonify,
    render_template,
    request,
    send_file
)

from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import RequestEntityTooLarge

from codigo.motor_gpu import MotorGPU


app = Flask(__name__)

# Limites de seguridad.
MAX_ARCHIVO = 40 * 1024 * 1024
MAX_DIMENSION = 6000
MAX_PIXELES = 36_000_000
HILOS = 256

app.config["MAX_CONTENT_LENGTH"] = MAX_ARCHIVO + 1024 * 1024

Image.MAX_IMAGE_PIXELS = MAX_PIXELES

FORMATOS = {"JPEG", "PNG", "PPM", "BMP"}

FILTROS = {
    "Media": (3, 9, 21),
    "Sharpen": (3,),
    "Laplaciano": (3,)
}

# Una solicitud GPU activa y una en espera.
capacidad = threading.BoundedSemaphore(2)
bloqueo_gpu = threading.Lock()

motor = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("practica3")


def obtener_motor():
    global motor

    if motor is None:
        motor = MotorGPU()

        logger.info(
            "GPU=%s | Inicializacion=%.3f ms | Compilacion=%.3f ms",
            motor.nombre_gpu,
            motor.inicializacion_ms,
            motor.compilacion_ms
        )

    return motor


def cerrar_motor():
    global motor

    with bloqueo_gpu:
        if motor is not None:
            motor.cerrar()
            motor = None


atexit.register(cerrar_motor)


def error_json(mensaje, estado, identificador=None):
    return jsonify({
        "estado": "error",
        "id": identificador,
        "mensaje": mensaje
    }), estado


@app.errorhandler(RequestEntityTooLarge)
def archivo_demasiado_grande(error):
    return error_json(
        "La solicitud supera el limite permitido de archivo",
        413,
        uuid.uuid4().hex
    )


def cargar_imagen(archivo):

    contenido = archivo.read(MAX_ARCHIVO + 1)

    if len(contenido) > MAX_ARCHIVO:
        raise RequestEntityTooLarge()

    if not contenido:
        raise ValueError("El archivo esta vacio")

    try:
        with Image.open(io.BytesIO(contenido)) as imagen:

            if imagen.format not in FORMATOS:
                raise ValueError("Formato de imagen no permitido")

            ancho, alto = imagen.size

            if (
                ancho < 1
                or alto < 1
                or ancho > MAX_DIMENSION
                or alto > MAX_DIMENSION
                or ancho * alto > MAX_PIXELES
            ):
                raise ValueError(
                    "Dimensiones no permitidas. Maximo 6000x6000"
                )

            imagen.verify()

        with Image.open(io.BytesIO(contenido)) as imagen:

            imagen.load()

            # Mantener el tratamiento de Practica 02.
            gris = imagen.convert("L")

            entrada = np.ascontiguousarray(
                np.asarray(gris),
                dtype=np.float32
            )

        return entrada

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError
    ) as exc:
        raise ValueError(
            "Archivo de imagen invalido o demasiado grande"
        ) from exc


def codificar_png(matriz):

    # Recorte exclusivamente para visualizacion.
    pixeles = np.clip(
        matriz, 0, 255
    ).astype(np.uint8)

    buffer = io.BytesIO()

    Image.fromarray(pixeles).save(
        buffer,
        format="PNG"
    )

    return buffer.getvalue()


@app.get("/")
def inicio():
    return render_template("index.html")


@app.get("/api/salud")
def salud():

    if not bloqueo_gpu.acquire(blocking=False):
        return jsonify({
            "estado": "ocupado"
        }), 503

    try:
        gpu = obtener_motor()

        return jsonify({
            "estado": "correcto",
            "gpu": gpu.nombre_gpu,
            "inicializacion_ms": gpu.inicializacion_ms,
            "compilacion_ms": gpu.compilacion_ms,
            "hilos": HILOS
        })

    except Exception:
        logger.exception("Error de inicializacion CUDA")

        return error_json(
            "No se pudo inicializar PyCUDA o acceder a la GPU",
            503
        )

    finally:
        bloqueo_gpu.release()


@app.post("/api/procesar")
def procesar():

    identificador = uuid.uuid4().hex

    inicio_solicitud = time.perf_counter()

    if not capacidad.acquire(blocking=False):
        return error_json(
            "Servidor ocupado. Intente nuevamente",
            429,
            identificador
        )

    try:
        inicio_espera = time.perf_counter()

        with bloqueo_gpu:

            espera_ms = (
                time.perf_counter() - inicio_espera
            ) * 1000

            inicio_servidor = time.perf_counter()

            archivo = request.files.get("imagen")

            if archivo is None:
                return error_json(
                    "Debe cargar una imagen",
                    400,
                    identificador
                )

            filtro = request.form.get("filtro", "")

            if filtro not in FILTROS:
                return error_json(
                    "Filtro desconocido",
                    400,
                    identificador
                )

            try:
                mascara = int(
                    request.form.get("mascara", "3")
                )
            except ValueError:
                return error_json(
                    "Mascara invalida",
                    400,
                    identificador
                )

            if mascara not in FILTROS[filtro]:
                return error_json(
                    "Tamano de mascara no permitido",
                    400,
                    identificador
                )

            inicio_preparacion = time.perf_counter()

            entrada = cargar_imagen(archivo)

            preparacion_ms = (
                time.perf_counter() - inicio_preparacion
            ) * 1000

            arranque_frio = motor is None

            inicio_gpu = time.perf_counter()

            gpu = obtener_motor()

            arranque_ms = (
                (time.perf_counter() - inicio_gpu) * 1000
                if arranque_frio else 0.0
            )

            resultado, tiempos = gpu.procesar(
                entrada,
                filtro,
                mascara,
                hilos=HILOS
            )

            inicio_salida = time.perf_counter()

            resultado_png = codificar_png(resultado)

            formato = request.form.get("salida", "png")

            if formato == "npy":
                # Resultado float32 sin recorte.
                buffer = io.BytesIO()
                np.save(buffer, resultado)

                datos = buffer.getvalue()
                extension = "npy"
                mime = "application/octet-stream"

            elif formato == "zip":
                # Evidencia visual y numerica.
                numerico = io.BytesIO()
                np.save(numerico, resultado)

                buffer = io.BytesIO()

                with zipfile.ZipFile(
                    buffer, "w", zipfile.ZIP_STORED
                ) as paquete:

                    paquete.writestr(
                        "resultado.png",
                        resultado_png
                    )

                    paquete.writestr(
                        "resultado_float32.npy",
                        numerico.getvalue()
                    )

                datos = buffer.getvalue()
                extension = "zip"
                mime = "application/zip"

            elif formato == "png":
                datos = resultado_png
                extension = "png"
                mime = "image/png"

            else:
                return error_json(
                    "Formato de salida desconocido",
                    400,
                    identificador
                )

            salida_ms = (
                time.perf_counter() - inicio_salida
            ) * 1000

            servidor_ms = (
                time.perf_counter() - inicio_servidor
            ) * 1000

            solicitud_ms = (
                time.perf_counter() - inicio_solicitud
            ) * 1000

            nombre = (
                f"{identificador}_{filtro}_"
                f"{mascara}x{mascara}.{extension}"
            )

            respuesta = send_file(
                io.BytesIO(datos),
                mimetype=mime,
                as_attachment=True,
                download_name=nombre,
                max_age=0
            )

            respuesta.headers["Cache-Control"] = "no-store"

            respuesta.headers["X-Request-ID"] = identificador

            respuesta.headers["X-Filtro"] = filtro
            respuesta.headers["X-Mascara"] = str(mascara)

            respuesta.headers["X-Espera-ms"] = str(espera_ms)
            respuesta.headers["X-Preparacion-ms"] = str(
                preparacion_ms
            )

            respuesta.headers["X-Salida-ms"] = str(salida_ms)
            respuesta.headers["X-Servidor-ms"] = str(servidor_ms)
            respuesta.headers["X-Solicitud-ms"] = str(solicitud_ms)
            respuesta.headers["X-Arranque-ms"] = str(arranque_ms)

            respuesta.headers["X-Arranque-Frio"] = (
                "1" if arranque_frio else "0"
            )

            for clave, valor in tiempos.items():
                respuesta.headers[
                    "X-" + clave.replace("_", "-")
                ] = str(valor)

            logger.info(
                "ID=%s | GPU=%s | Filtro=%s | Mascara=%s | "
                "Resolucion=%sx%s | Kernel=%.3f ms | "
                "Servidor=%.3f ms | Espera=%.3f ms",
                identificador,
                gpu.nombre_gpu,
                filtro,
                mascara,
                tiempos["ancho"],
                tiempos["alto"],
                tiempos["kernel_ms"],
                servidor_ms,
                espera_ms
            )

            return respuesta

    except RequestEntityTooLarge:
        return error_json(
            "El archivo supera el limite de 40 MiB",
            413,
            identificador
        )

    except ValueError as exc:
        return error_json(
            str(exc),
            400,
            identificador
        )

    except MemoryError:
        logger.exception("Memoria insuficiente: %s", identificador)

        return error_json(
            "Memoria insuficiente para procesar la imagen",
            503,
            identificador
        )

    except Exception:
        logger.exception(
            "Error de procesamiento GPU: %s",
            identificador
        )

        return error_json(
            "Error de procesamiento PyCUDA. Revise los registros",
            503,
            identificador
        )

    finally:
        capacidad.release()


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=False,
        use_reloader=False,
        threaded=True
    )
