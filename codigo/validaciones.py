"""Validaciones manuales, bordes e imagen irregular CPU/PyCUDA."""
import argparse
import numpy as np
from filtros_python import crear_mascara, convolucion_python
from filtros_pycuda import FiltroPyCUDA

TOLERANCIA = 1e-4


def validar(referencia, actual, descripcion):
    if referencia.shape != actual.shape:
        raise AssertionError(f'{descripcion}: dimensiones distintas')
    error = np.abs(referencia.astype(np.float64) - actual.astype(np.float64))
    maximo = float(error.max())
    medio = float(error.mean())
    correcto = bool(np.isfinite(actual).all() and maximo <= TOLERANCIA)
    print(f'{descripcion}: {"CORRECTO" if correcto else "ERROR"} | máximo={maximo:.8f} | medio={medio:.8f}')
    if not correcto:
        raise AssertionError(f'Falló: {descripcion}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sin-gpu', action='store_true')
    args = parser.parse_args()
    motor = None if args.sin_gpu else FiltroPyCUDA()

    # Patrón 5x5 conocido; esperado independiente calculado a mano.
    entrada = np.arange(1, 26, dtype=np.float32).reshape(5, 5)
    esperado_media3 = np.array([
        [16, 27, 33, 39, 28],
        [39, 63, 72, 81, 57],
        [69, 108, 117, 126, 87],
        [99, 153, 162, 171, 117],
        [76, 117, 123, 129, 88]
    ], dtype=np.float32) / np.float32(9.0)
    # Para otras máscaras, se contrastan los dos algoritmos y casos puntuales.
    media = crear_mascara('Media', 3)
    cpu = convolucion_python(entrada, media)
    validar(esperado_media3, cpu, 'Media 3x3 manual vs Python')

    for nombre, tam in [('Media', 3), ('Media', 9), ('Media', 21),
                        ('Sharpen', 3), ('Laplaciano', 3)]:
        mascara = crear_mascara(nombre, tam)
        cpu = convolucion_python(entrada, mascara)
        if motor is not None:
            gpu, _ = motor.ejecutar(entrada, mascara, hilos=128)
            validar(cpu, gpu, f'{nombre} {tam}x{tam} 5x5 CPU/GPU')
        print(f'Caso 5x5: {nombre} {tam}x{tam} OK')

    # Anchura no divisible por 128/256/512, con bordes reales.
    irregular = np.arange(37 * 53, dtype=np.float32).reshape(37, 53) % 251
    for nombre, tam in [('Media', 3), ('Sharpen', 3), ('Laplaciano', 3)]:
        mascara = crear_mascara(nombre, tam)
        cpu = convolucion_python(irregular, mascara)
        if motor is not None:
            for hilos in (128, 256, 512):
                gpu, _ = motor.ejecutar(irregular, mascara, hilos=hilos)
                validar(cpu, gpu, f'{nombre} irregular 53x37, {hilos} hilos')
    print('\nVALIDACIONES COMPLETADAS')


if __name__ == '__main__':
    main()