"""Prepara imágenes PGM. Preferir reutilizar los PGM de la Práctica 01."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
IMAGENES = RAIZ / 'imagenes'


def cargar_pgm(ruta):
    """Lee PGM binario P5 de 8 bits; admite comentarios y espacios en cabecera."""
    with open(ruta, 'rb') as f:
        def token():
            dato = bytearray()
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('Cabecera PGM incompleta')
                if c == b'#':
                    f.readline()
                    continue
                if c not in b' \t\r\n':
                    dato.extend(c)
                    break
            while True:
                c = f.read(1)
                if not c or c in b' \t\r\n':
                    break
                dato.extend(c)
            return bytes(dato)
        magic = token()
        if magic != b'P5':
            raise ValueError('Se requiere PGM binario P5')
        ancho, alto = int(token()), int(token())
        maximo = int(token())
        if maximo != 255:
            raise ValueError('Solo se admiten PGM P5 de 8 bits (maxval=255)')
        pixeles = f.read(ancho * alto)
        if len(pixeles) != ancho * alto:
            raise ValueError('Archivo PGM truncado')
    return np.frombuffer(pixeles, dtype=np.uint8).reshape(alto, ancho).copy()


def guardar_pgm(ruta, imagen):
    imagen = np.asarray(imagen, dtype=np.uint8)
    if imagen.ndim != 2:
        raise ValueError('Se requiere una imagen 2D')
    alto, ancho = imagen.shape
    with open(ruta, 'wb') as f:
        f.write(f'P5\n{ancho} {alto}\n255\n'.encode('ascii'))
        f.write(imagen.tobytes())


def preparar_desde_original(ruta_original):
    """Alternativa reproducible. Para comparación estricta usar PGM de Práctica 01."""
    with Image.open(ruta_original) as original:
        rgb = np.asarray(original.convert('RGB'), dtype=np.float32)
    h, w, _ = rgb.shape
    lado = min(h, w)
    y0, x0 = (h - lado) // 2, (w - lado) // 2
    rgb = rgb[y0:y0 + lado, x0:x0 + lado]
    gris = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    gris8 = np.clip(np.rint(gris), 0, 255).astype(np.uint8)
    IMAGENES.mkdir(parents=True, exist_ok=True)
    for lado_destino in (2000, 4000, 6000):
        imagen = Image.fromarray(gris8).resize((lado_destino, lado_destino), Image.Resampling.BILINEAR)
        destino = IMAGENES / f'imagen_{lado_destino}.pgm'
        guardar_pgm(destino, np.asarray(imagen))
        print('Creada:', destino)
    print('Aviso: el redimensionado Pillow puede diferir del C++ de Práctica 01.')


def main():
    p = argparse.ArgumentParser(description='Verifica o genera imágenes PGM')
    p.add_argument('--original', type=Path, help='JPG original para generar PGM (opcional)')
    args = p.parse_args()
    if args.original:
        preparar_desde_original(args.original)
    else:
        for lado in (2000, 4000, 6000):
            ruta = IMAGENES / f'imagen_{lado}.pgm'
            if ruta.exists():
                img = cargar_pgm(ruta)
                print(f'OK: {ruta.name} {img.shape[1]}x{img.shape[0]}')
            else:
                print('FALTA:', ruta)


if __name__ == '__main__':
    main()