
import csv
import statistics

from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from PIL import Image


# Directorios del proyecto.

RAIZ = Path(__file__).resolve().parents[1]

RESULTADOS = RAIZ / "resultados"
GRAFICAS = RAIZ / "graficas"
IMAGENES = RAIZ / "imagenes"

GRAFICAS.mkdir(parents=True, exist_ok=True)


# Configuracion visual.

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.20,
    "axes.axisbelow": True
})

AZUL = "#2563EB"
VERDE = "#16A34A"
NARANJA = "#EA580C"
MORADO = "#9333EA"
GRIS = "#64748B"

COLORES = [
    AZUL,
    VERDE,
    NARANJA,
    MORADO,
    GRIS
]


def leer_csv(nombre):

    ruta = RESULTADOS / nombre

    with open(
        ruta,
        "r",
        encoding="utf-8-sig",
        newline=""
    ) as archivo:

        return list(csv.DictReader(archivo))


def guardar(nombre):

    ruta = GRAFICAS / nombre

    plt.tight_layout()

    plt.savefig(
        ruta,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print("Generada:", ruta.name)


def valores(filas, campo):

    return [
        float(fila[campo])
        for fila in filas
        if fila.get(campo, "") != ""
    ]


def mediana(filas, campo):

    datos = valores(filas, campo)

    return statistics.median(datos)


def desviacion(filas, campo):

    datos = valores(filas, campo)

    if len(datos) > 1:
        return statistics.stdev(datos)

    return 0.0


def etiquetas_barras(ax, barras, decimales=2):

    ax.bar_label(
        barras,
        fmt=f"%.{decimales}f",
        padding=4,
        fontsize=10
    )

    ax.margins(y=0.18)


def barras_simples(
    nombres,
    datos,
    titulo,
    ylabel,
    archivo,
    colores=None,
    errores=None,
    decimales=2
):

    fig, ax = plt.subplots()

    barras = ax.bar(
        nombres,
        datos,
        color=colores or AZUL,
        yerr=errores,
        capsize=5 if errores is not None else 0
    )

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)

    etiquetas_barras(
        ax,
        barras,
        decimales
    )

    guardar(archivo)


def barras_agrupadas(
    nombres,
    serie_a,
    serie_b,
    nombre_a,
    nombre_b,
    titulo,
    ylabel,
    archivo
):

    x = np.arange(len(nombres))

    ancho = 0.35

    fig, ax = plt.subplots()

    barras_a = ax.bar(
        x - ancho / 2,
        serie_a,
        ancho,
        label=nombre_a,
        color=AZUL
    )

    barras_b = ax.bar(
        x + ancho / 2,
        serie_b,
        ancho,
        label=nombre_b,
        color=VERDE
    )

    ax.set_xticks(x)
    ax.set_xticklabels(nombres)

    ax.set_title(titulo)
    ax.set_ylabel(ylabel)

    ax.legend()

    etiquetas_barras(ax, barras_a)
    etiquetas_barras(ax, barras_b)

    guardar(archivo)


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
        encoding="utf-8",
        newline=""
    ) as archivo:

        escritor = csv.DictWriter(
            archivo,
            fieldnames=campos
        )

        escritor.writeheader()
        escritor.writerows(filas)

    print("CSV generado:", ruta.name)


def experimento_e1(windows, docker, web):

    servicio = [
        r for r in web
        if r["experimento"] == "E1"
    ]

    grupos = [
        windows,
        docker,
        servicio
    ]

    nombres = [
        "Windows directo",
        "Docker directo",
        "Servicio web"
    ]

    modos = [
        "windows",
        "docker_directo",
        "servicio_web"
    ]

    # Grafica 1: Kernel por modalidad.

    barras_simples(
        nombres,
        [mediana(g, "kernel_ms") for g in grupos],
        "E1. Tiempo del kernel por modalidad",
        "Tiempo (ms)",
        "01_kernel_modalidades.png",
        COLORES[:3]
    )

    # Grafica 2: Tiempo total.

    totales = [
        mediana(windows, "total_directo_ms"),
        mediana(docker, "total_directo_ms"),
        mediana(servicio, "cliente_ms")
    ]

    barras_simples(
        nombres,
        totales,
        "E1. Tiempo total por modalidad",
        "Tiempo (ms)",
        "02_total_modalidades.png",
        COLORES[:3]
    )

    # Grafica 3: Transferencias y kernel.

    fig, ax = plt.subplots()

    x = np.arange(len(nombres))

    ancho = 0.25

    campos = [
        ("h2d_ms", "H2D", AZUL),
        ("kernel_ms", "Kernel", VERDE),
        ("d2h_ms", "D2H", NARANJA)
    ]

    for i, (campo, etiqueta, color) in enumerate(campos):

        datos = [
            mediana(g, campo)
            for g in grupos
        ]

        barras = ax.bar(
            x + (i - 1) * ancho,
            datos,
            ancho,
            label=etiqueta,
            color=color
        )

        etiquetas_barras(ax, barras)

    ax.set_xticks(x)
    ax.set_xticklabels(nombres)

    ax.set_title(
        "E1. Transferencias y procesamiento GPU"
    )

    ax.set_ylabel("Tiempo (ms)")
    ax.legend()

    guardar("03_transferencias_kernel.png")

    # Grafica 15: Variabilidad del kernel.

    fig, ax = plt.subplots()

    ax.boxplot(
        [valores(g, "kernel_ms") for g in grupos]
    )

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(nombres)

    ax.set_title(
        "E1. Variabilidad del tiempo del kernel"
    )

    ax.set_ylabel("Tiempo (ms)")

    guardar("15_variabilidad_kernel.png")

    # Resumen estadistico de E1.

    resumen = []

    for modo, grupo in zip(modos, grupos):

        fila = {
            "experimento": "E1",
            "modo": modo,
            "resolucion": 4000,
            "filtro": "Media",
            "mascara": 9,
            "hilos": 256,
            "repeticiones": len(grupo)
        }

        campos = [
            "preparacion_ms",
            "reserva_ms",
            "h2d_ms",
            "kernel_ms",
            "d2h_ms",
            "gpu_total_ms",
            "motor_total_ms",
            "salida_ms",
            "total_directo_ms",
            "servidor_ms",
            "cliente_ms",
            "bytes_resultado"
        ]

        for campo in campos:

            if valores(grupo, campo):

                fila[campo + "_mediana"] = (
                    mediana(grupo, campo)
                )

                fila[campo + "_desviacion"] = (
                    desviacion(grupo, campo)
                )

        resumen.append(fila)

    guardar_csv(
        "resumen_modalidades.csv",
        resumen
    )


def experimento_e2(resumen):

    filas = [
        r for r in resumen
        if r["experimento"] == "E2"
    ]

    filas.sort(
        key=lambda r: int(r["resolucion"])
    )

    nombres = [
        f"{r['resolucion']}x{r['resolucion']}"
        for r in filas
    ]

    kernel = valores(
        filas, "kernel_ms_mediana"
    )

    servidor = valores(
        filas, "servidor_ms_mediana"
    )

    cliente = valores(
        filas, "cliente_ms_mediana"
    )

    # Grafica 4.

    fig, ax = plt.subplots()

    ax.errorbar(
        nombres,
        kernel,
        yerr=valores(
            filas, "kernel_ms_desviacion"
        ),
        marker="o",
        linewidth=2,
        capsize=5,
        color=AZUL
    )

    ax.set_title(
        "E2. Tiempo del kernel segun resolucion"
    )

    ax.set_xlabel("Resolucion de imagen")
    ax.set_ylabel("Tiempo (ms)")

    guardar("04_kernel_resolucion.png")

    # Grafica 5.

    fig, ax = plt.subplots()

    ax.plot(
        nombres,
        servidor,
        marker="o",
        linewidth=2,
        label="Servidor",
        color=AZUL
    )

    ax.plot(
        nombres,
        cliente,
        marker="s",
        linewidth=2,
        label="Cliente",
        color=VERDE
    )

    ax.set_title(
        "E2. Tiempo total segun resolucion"
    )

    ax.set_xlabel("Resolucion de imagen")
    ax.set_ylabel("Tiempo (ms)")
    ax.legend()

    guardar("05_total_resolucion.png")


def experimento_e3(resumen, web):

    filas = [
        r for r in resumen
        if r["experimento"] == "E3"
    ]

    nombres = [
        f"{r['filtro']}\n{r['mascara']}x{r['mascara']}"
        for r in filas
    ]

    # Grafica 6.

    barras_simples(
        nombres,
        valores(filas, "kernel_ms_mediana"),
        "E3. Tiempo del kernel por filtro",
        "Tiempo (ms)",
        "06_kernel_filtros.png",
        COLORES,
        errores=valores(
            filas, "kernel_ms_desviacion"
        )
    )

    # Grafica 7.

    barras_agrupadas(
        nombres,
        valores(filas, "servidor_ms_mediana"),
        valores(filas, "cliente_ms_mediana"),
        "Servidor",
        "Cliente",
        "E3. Tiempo total por filtro y mascara",
        "Tiempo (ms)",
        "07_total_filtros.png"
    )

    # Grafica 8:
    # Descomposicion por solicitud antes de
    # calcular las medianas de los componentes.

    grupos = []

    for fila in filas:

        grupo = [
            r for r in web
            if r["experimento"] == "E3"
            and r["filtro"] == fila["filtro"]
            and r["mascara"] == fila["mascara"]
        ]

        grupos.append(grupo)

    componentes = [
        ("preparacion_ms", "Preparacion", AZUL),
        ("reserva_ms", "Reserva GPU", GRIS),
        ("h2d_ms", "H2D", MORADO),
        ("kernel_ms", "Kernel", VERDE),
        ("d2h_ms", "D2H", "#0891B2"),
        ("salida_ms", "Codificacion PNG", NARANJA)
    ]

    fig, ax = plt.subplots()

    inferior = np.zeros(len(nombres))

    for campo, etiqueta, color in componentes:

        datos = np.array([
            mediana(g, campo)
            for g in grupos
        ])

        ax.bar(
            nombres,
            datos,
            bottom=inferior,
            label=etiqueta,
            color=color
        )

        inferior += datos

    ax.set_title(
        "E3. Medianas de los componentes del servidor"
    )

    ax.set_ylabel("Tiempo (ms)")

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3
    )

    guardar("08_componentes_servidor.png")

    # Grafica 14:
    # Codificacion PNG y tamano del resultado.

    tiempos = [
        mediana(g, "salida_ms")
        for g in grupos
    ]

    tamanos = [
        mediana(g, "bytes_resultado") / (1024 ** 2)
        for g in grupos
    ]

    fig, ax1 = plt.subplots()

    barras = ax1.bar(
        nombres,
        tiempos,
        color=AZUL,
        alpha=0.8,
        label="Codificacion PNG"
    )

    ax1.set_ylabel(
        "Tiempo de codificacion (ms)",
        color=AZUL
    )

    etiquetas_barras(ax1, barras)

    ax2 = ax1.twinx()

    ax2.plot(
        nombres,
        tamanos,
        marker="o",
        linewidth=2,
        color=NARANJA,
        label="Tamano PNG"
    )

    ax2.set_ylabel(
        "Tamano del resultado (MiB)",
        color=NARANJA
    )

    ax2.grid(False)

    ax1.set_title(
        "E3. Codificacion y tamano de salida PNG"
    )

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    guardar("14_codificacion_png.png")

    resumen_salida = []

    for fila, tiempo, tamano in zip(
        filas, tiempos, tamanos
    ):

        resumen_salida.append({
            "filtro": fila["filtro"],
            "mascara": fila["mascara"],
            "salida_ms_mediana": tiempo,
            "tamano_png_mib_mediana": tamano
        })

    guardar_csv(
        "resumen_codificacion.csv",
        resumen_salida
    )


def experimento_e4(resumen, concurrencia):

    nombres = [
        f"{r['concurrencia']} cliente(s)"
        for r in resumen
    ]

    # Grafica 9.

    barras_simples(
        nombres,
        valores(resumen, "cliente_ms_mediana"),
        "E4. Latencia segun concurrencia",
        "Latencia (ms)",
        "09_latencia_concurrencia.png",
        [AZUL, NARANJA],
        errores=valores(
            resumen, "cliente_ms_desviacion"
        )
    )

    # Grafica 10.

    barras_simples(
        nombres,
        valores(resumen, "espera_ms_mediana"),
        "E4. Tiempo de espera segun concurrencia",
        "Tiempo de espera (ms)",
        "10_espera_concurrencia.png",
        [VERDE, NARANJA],
        decimales=3
    )

    # Grafica 11.

    barras_simples(
        nombres,
        valores(resumen, "throughput_mediana"),
        "E4. Rendimiento segun concurrencia",
        "Solicitudes exitosas por segundo",
        "11_throughput_concurrencia.png",
        [AZUL, VERDE],
        errores=valores(
            resumen, "throughput_desviacion"
        ),
        decimales=3
    )

    # Grafica 12.

    fig, ax = plt.subplots()

    configuraciones = [
        ("1", "1", "Un cliente", AZUL),
        ("2", "1", "Dos clientes - A", VERDE),
        ("2", "2", "Dos clientes - B", NARANJA)
    ]

    for cantidad, cliente, etiqueta, color in configuraciones:

        filas = [
            r for r in concurrencia
            if r["concurrencia"] == cantidad
            and r["cliente"] == cliente
        ]

        filas.sort(
            key=lambda r: int(r["ronda"])
        )

        ax.plot(
            [int(r["ronda"]) for r in filas],
            valores(filas, "cliente_ms"),
            marker="o",
            linewidth=2,
            label=etiqueta,
            color=color
        )

    ax.set_xticks([1, 2, 3, 4, 5])

    ax.set_title(
        "E4. Latencia individual por ronda"
    )

    ax.set_xlabel("Ronda")
    ax.set_ylabel("Latencia (ms)")
    ax.legend()

    guardar("12_latencia_por_ronda.png")


def comparacion_visual():

    archivos = [
        (
            IMAGENES / "imagen_4000.pgm",
            "Imagen original"
        ),
        (
            RESULTADOS / "Media_3x3_4000.png",
            "Media 3x3"
        ),
        (
            RESULTADOS / "Media_9x9_4000.png",
            "Media 9x9"
        ),
        (
            RESULTADOS / "Media_21x21_4000.png",
            "Media 21x21"
        ),
        (
            RESULTADOS / "Sharpen_3x3_4000.png",
            "Sharpen 3x3"
        ),
        (
            RESULTADOS / "Laplaciano_3x3_4000.png",
            "Laplaciano 3x3"
        )
    ]

    # Misma region espacial para todas las imagenes.

    x = 1744
    y = 1744
    lado = 512

    region = (
        x,
        y,
        x + lado,
        y + lado
    )

    fig, ejes = plt.subplots(
        2,
        3,
        figsize=(13, 9)
    )

    for ax, (ruta, titulo) in zip(
        ejes.flat, archivos
    ):

        with Image.open(ruta) as imagen:

            recorte = np.asarray(
                imagen.convert("L").crop(region)
            )

        ax.imshow(
            recorte,
            cmap="gray",
            vmin=0,
            vmax=255
        )

        ax.set_title(titulo)
        ax.axis("off")

    fig.suptitle(
        "Comparacion visual de filtros de convolucion",
        fontsize=16
    )

    guardar("13_comparacion_visual.png")


def main():

    print("=" * 65)
    print("GENERACION DE GRAFICAS - PRACTICA 03")
    print("=" * 65)

    windows = [
        r for r in leer_csv(
            "mediciones_windows.csv"
        )
        if r["estado"] == "CORRECTO"
    ]

    docker = [
        r for r in leer_csv(
            "mediciones_docker_directo.csv"
        )
        if r["estado"] == "CORRECTO"
    ]

    web = [
        r for r in leer_csv(
            "mediciones_web.csv"
        )
        if r["estado"] == "CORRECTO"
    ]

    resumen_web = leer_csv(
        "resumen_web.csv"
    )

    concurrencia = [
        r for r in leer_csv(
            "concurrencia.csv"
        )
        if r["estado"] == "CORRECTO"
    ]

    resumen_concurrencia = leer_csv(
        "resumen_concurrencia.csv"
    )

    print("Mediciones Windows:", len(windows))
    print("Mediciones Docker:", len(docker))
    print("Mediciones Web:", len(web))
    print("Mediciones E4:", len(concurrencia))

    if (
        len(windows) != 5
        or len(docker) != 5
        or len(web) != 45
        or len(concurrencia) != 15
    ):

        raise RuntimeError(
            "Las mediciones estan incompletas"
        )

    experimento_e1(
        windows,
        docker,
        web
    )

    experimento_e2(resumen_web)

    experimento_e3(
        resumen_web,
        web
    )

    experimento_e4(
        resumen_concurrencia,
        concurrencia
    )

    comparacion_visual()

    print("\n" + "=" * 65)
    print("GENERACION FINALIZADA")
    print("=" * 65)

    print("Total mediciones:", 70)
    print("Graficas generadas:", 15)
    print("Directorio:", GRAFICAS)


if __name__ == "__main__":
    main()
