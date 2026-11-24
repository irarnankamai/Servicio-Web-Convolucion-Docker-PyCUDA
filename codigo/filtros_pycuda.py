"""Convolución CUDA C propia compilada y lanzada mediante PyCUDA."""
import time
import numpy as np

CUDA_C = r'''
extern "C" __global__ void convolucion(
    const float *entrada, float *salida,
    const int ancho, const int alto,
    const float *mascara, const int k)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y;
    if (x >= ancho || y >= alto) return;
    const int radio = k / 2;
    float suma = 0.0f;
    for (int j = 0; j < k; ++j) {
        const int yy = y + j - radio;
        if (yy < 0 || yy >= alto) continue;
        for (int i = 0; i < k; ++i) {
            const int xx = x + i - radio;
            if (xx >= 0 && xx < ancho) {
                suma += entrada[yy * ancho + xx] * mascara[j * k + i];
            }
        }
    }
    salida[y * ancho + x] = suma;
}
'''


def _modulos():
    # Importación diferida: los tests de Python funcionan incluso sin PyCUDA.
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401; inicializa contexto CUDA
    from pycuda.compiler import SourceModule
    return cuda, SourceModule


class FiltroPyCUDA:
    def __init__(self):
        cuda, SourceModule = _modulos()
        self.cuda = cuda
        inicio = time.perf_counter()
        self.modulo = SourceModule(CUDA_C, options=['--fmad=false'], no_extern_c=True)
        self.kernel = self.modulo.get_function('convolucion')
        self.compilacion_ms = (time.perf_counter() - inicio) * 1000.0

    def ejecutar(self, entrada, mascara, hilos=256, repeticiones=1):
        if hilos not in (128, 256, 512):
            raise ValueError('Hilos válidos: 128, 256, 512')
        if repeticiones < 1:
            raise ValueError('repeticiones debe ser >= 1')
        entrada = np.ascontiguousarray(entrada, dtype=np.float32)
        mascara = np.ascontiguousarray(mascara, dtype=np.float32)
        if entrada.ndim != 2 or mascara.ndim != 2 or mascara.shape[0] != mascara.shape[1] or mascara.shape[0] % 2 == 0:
            raise ValueError('Imagen 2D y máscara cuadrada impar requeridas')
        alto, ancho = entrada.shape
        k = mascara.shape[0]
        salida = np.empty_like(entrada)
        cuda = self.cuda
        d_entrada = d_salida = d_mascara = None
        registros = []
        try:
            # Asignación excluida de los tiempos de transferencia y GPU total.
            d_entrada = cuda.mem_alloc(entrada.nbytes)
            d_salida = cuda.mem_alloc(salida.nbytes)
            d_mascara = cuda.mem_alloc(mascara.nbytes)
            grid = ((ancho + hilos - 1) // hilos, alto, 1)
            block = (hilos, 1, 1)

            def lanzar():
                self.kernel(d_entrada, d_salida, np.int32(ancho), np.int32(alto),
                            d_mascara, np.int32(k), block=block, grid=grid)

            # Calentamiento: incluye las primeras transferencias y ejecución.
            cuda.memcpy_htod(d_entrada, entrada)
            cuda.memcpy_htod(d_mascara, mascara)
            lanzar()
            cuda.Context.synchronize()

            for repeticion in range(1, repeticiones + 1):
                cuda.Context.synchronize()
                t_total = time.perf_counter()
                t = time.perf_counter()
                cuda.memcpy_htod(d_entrada, entrada)
                cuda.memcpy_htod(d_mascara, mascara)
                cuda.Context.synchronize()
                h2d = (time.perf_counter() - t) * 1000.0

                ev_inicio = cuda.Event()
                ev_fin = cuda.Event()
                ev_inicio.record()
                lanzar()
                ev_fin.record()
                ev_fin.synchronize()
                kernel_ms = ev_inicio.time_till(ev_fin)

                t = time.perf_counter()
                cuda.memcpy_dtoh(salida, d_salida)
                cuda.Context.synchronize()
                d2h = (time.perf_counter() - t) * 1000.0
                total = (time.perf_counter() - t_total) * 1000.0
                registros.append(dict(repeticion=repeticion, h2d_ms=h2d,
                                      kernel_ms=kernel_ms, d2h_ms=d2h,
                                      gpu_total_ms=total, bloques=grid[0] * grid[1]))
        finally:
            for buf in (d_entrada, d_salida, d_mascara):
                if buf is not None:
                    buf.free()
        return salida, registros


def convolucion_pycuda(entrada, mascara, hilos=256, motor=None):
    motor = motor or FiltroPyCUDA()
    salida, _ = motor.ejecutar(entrada, mascara, hilos=hilos, repeticiones=1)
    return salida
