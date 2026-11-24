"""Filtros secuenciales, con bucles explícitos y bordes en cero."""
import numpy as np


def crear_mascara(nombre: str, tamano: int = 3) -> np.ndarray:
    nombre = nombre.strip().lower()
    if nombre == 'media':
        if tamano not in (3, 9, 21):
            raise ValueError('Media solo admite tamaños 3, 9 y 21')
        return np.full((tamano, tamano), np.float32(1.0 / (tamano * tamano)), dtype=np.float32)
    if tamano != 3:
        raise ValueError('Sharpen y Laplaciano solo admiten tamaño 3')
    if nombre == 'sharpen':
        return np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    if nombre in ('laplaciano', 'laplacian'):
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    raise ValueError(f'Filtro desconocido: {nombre}')


def convolucion_python(entrada: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Convolución/correlación simétrica de máscaras dadas; no usa filtros de bibliotecas.

    Las máscaras empleadas son simétricas, por lo que correlación y convolución coinciden.
    Operaciones explícitas float32 para acercarse al kernel CUDA sin FMA.
    """
    entrada = np.ascontiguousarray(entrada, dtype=np.float32)
    mascara = np.ascontiguousarray(mascara, dtype=np.float32)
    if entrada.ndim != 2 or mascara.ndim != 2 or mascara.shape[0] != mascara.shape[1] or mascara.shape[0] % 2 == 0:
        raise ValueError('Se espera imagen 2D y máscara cuadrada impar')
    alto, ancho = entrada.shape
    k = mascara.shape[0]
    radio = k // 2
    salida = np.zeros((alto, ancho), dtype=np.float32)
    for y in range(alto):
        for x in range(ancho):
            suma = np.float32(0.0)
            for j in range(k):
                yy = y + j - radio
                if yy < 0 or yy >= alto:
                    continue
                for i in range(k):
                    xx = x + i - radio
                    if 0 <= xx < ancho:
                        producto = np.float32(entrada[yy, xx] * mascara[j, i])
                        suma = np.float32(suma + producto)
            salida[y, x] = suma
    return salida