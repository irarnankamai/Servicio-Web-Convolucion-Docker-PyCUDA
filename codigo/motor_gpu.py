
import time
import numpy as np

from codigo.filtros_pycuda import CUDA_C
from codigo.filtros_python import crear_mascara


class MotorGPU:

    def __init__(self):

        inicio = time.perf_counter()

        import pycuda.driver as cuda
        from pycuda.compiler import SourceModule

        self.cuda = cuda
        self.contexto = None
        self.modulo = None
        self.kernel = None

        cuda.init()

        if cuda.Device.count() == 0:
            raise RuntimeError("No se encontro una GPU NVIDIA")

        dispositivo = cuda.Device(0)

        self.nombre_gpu = dispositivo.name()

        self.contexto = dispositivo.make_context()

        try:

            inicio_compilacion = time.perf_counter()

            self.modulo = SourceModule(
                CUDA_C,
                options=["--fmad=false"],
                no_extern_c=True
            )

            self.kernel = self.modulo.get_function(
                "convolucion"
            )

            self.compilacion_ms = (
                time.perf_counter() - inicio_compilacion
            ) * 1000

            self.inicializacion_ms = (
                time.perf_counter() - inicio
            ) * 1000

        except Exception:

            self.contexto.pop()
            self.contexto.detach()
            self.contexto = None

            raise

        self.contexto.pop()


    def procesar(self, entrada, filtro, tamano, hilos=256):

        if self.contexto is None:
            raise RuntimeError("Motor GPU cerrado")

        if hilos not in (128, 256, 512):
            raise ValueError("Cantidad de hilos no permitida")

        if filtro not in ("Media", "Sharpen", "Laplaciano"):
            raise ValueError("Filtro desconocido")

        permitidos = (
            (3, 9, 21)
            if filtro == "Media"
            else (3,)
        )

        if tamano not in permitidos:
            raise ValueError("Tamano de mascara no permitido")

        inicio_total = time.perf_counter()

        entrada = np.ascontiguousarray(
            entrada,
            dtype=np.float32
        )

        if entrada.ndim != 2 or entrada.size == 0:
            raise ValueError("Se requiere una imagen 2D valida")

        mascara = np.ascontiguousarray(
            crear_mascara(filtro, tamano),
            dtype=np.float32
        )

        alto, ancho = entrada.shape

        salida = np.empty_like(entrada)

        grid = (
            (ancho + hilos - 1) // hilos,
            alto,
            1
        )

        block = (hilos, 1, 1)

        cuda = self.cuda

        d_entrada = None
        d_salida = None
        d_mascara = None

        evento_inicio = None
        evento_fin = None

        # Activar el contexto CUDA.
        self.contexto.push()

        try:

            libre, total = cuda.mem_get_info()

            requerida = (
                entrada.nbytes
                + salida.nbytes
                + mascara.nbytes
            )

            if requerida > libre:
                raise MemoryError("Memoria GPU insuficiente")

            # Reserva de memoria GPU.
            inicio = time.perf_counter()

            d_entrada = cuda.mem_alloc(entrada.nbytes)
            d_salida = cuda.mem_alloc(salida.nbytes)
            d_mascara = cuda.mem_alloc(mascara.nbytes)

            reserva_ms = (
                time.perf_counter() - inicio
            ) * 1000

            inicio_gpu = time.perf_counter()

            # Transferencia CPU hacia GPU.
            inicio = time.perf_counter()

            cuda.memcpy_htod(d_entrada, entrada)
            cuda.memcpy_htod(d_mascara, mascara)

            cuda.Context.synchronize()

            h2d_ms = (
                time.perf_counter() - inicio
            ) * 1000

            # Medicion del kernel mediante eventos CUDA.
            evento_inicio = cuda.Event()
            evento_fin = cuda.Event()

            evento_inicio.record()

            self.kernel(
                d_entrada,
                d_salida,
                np.int32(ancho),
                np.int32(alto),
                d_mascara,
                np.int32(tamano),
                block=block,
                grid=grid
            )

            evento_fin.record()
            evento_fin.synchronize()

            kernel_ms = evento_inicio.time_till(
                evento_fin
            )

            # Transferencia GPU hacia CPU.
            inicio = time.perf_counter()

            cuda.memcpy_dtoh(salida, d_salida)

            cuda.Context.synchronize()

            d2h_ms = (
                time.perf_counter() - inicio
            ) * 1000

            gpu_total_ms = (
                time.perf_counter() - inicio_gpu
            ) * 1000

        finally:

            # Liberar eventos y memoria antes de
            # desactivar el contexto CUDA.
            try:

                evento_inicio = None
                evento_fin = None

                for memoria in (
                    d_entrada,
                    d_salida,
                    d_mascara
                ):

                    if memoria is not None:
                        memoria.free()

            finally:

                self.contexto.pop()

        # Validar valores numericos.
        if not np.all(np.isfinite(salida)):
            raise RuntimeError(
                "El resultado contiene valores no validos"
            )

        motor_total_ms = (
            time.perf_counter() - inicio_total
        ) * 1000

        tiempos = {

            "reserva_ms": reserva_ms,
            "h2d_ms": h2d_ms,
            "kernel_ms": kernel_ms,
            "d2h_ms": d2h_ms,
            "gpu_total_ms": gpu_total_ms,
            "motor_total_ms": motor_total_ms,

            "ancho": ancho,
            "alto": alto,

            "hilos": hilos,

            "bloques_x": grid[0],
            "bloques_y": grid[1],

            "bloques": grid[0] * grid[1],

            "gpu": self.nombre_gpu
        }

        return salida, tiempos


    def cerrar(self):

        if self.contexto is not None:

            self.contexto.push()

            try:

                self.kernel = None
                self.modulo = None

            finally:

                self.contexto.pop()
                self.contexto.detach()

                self.contexto = None
