
"use strict";

const obtener = id => document.getElementById(id);

const formulario = obtener("formulario");
const imagen = obtener("imagen");
const filtro = obtener("filtro");
const mascara = obtener("mascara");
const boton = obtener("procesar");

const MAX_ARCHIVO = 40 * 1024 * 1024;

let urlOriginal = null;
let urlResultado = null;
let procesando = false;

const metricas = {
    h2d: "X-h2d-ms",
    kernel: "X-kernel-ms",
    d2h: "X-d2h-ms",
    totalGPU: "X-gpu-total-ms",
    preparacion: "X-Preparacion-ms",
    reserva: "X-reserva-ms",
    salida: "X-Salida-ms",
    espera: "X-Espera-ms",
    servidor: "X-Servidor-ms"
};


function mostrarMensaje(texto, tipo) {
    obtener("mensaje").textContent = texto;
    obtener("mensaje").className = tipo;
}


function limpiarResultado() {

    if (urlResultado) {
        URL.revokeObjectURL(urlResultado);
        urlResultado = null;
    }

    obtener("resultado").removeAttribute("src");
    obtener("resultado").hidden = true;
    obtener("textoResultado").hidden = false;

    obtener("descargar").hidden = true;
    obtener("descargar").removeAttribute("href");

    Object.keys(metricas).forEach(id => {
        obtener(id).textContent = "—";
    });

    [
        "cliente",
        "identificador",
        "gpu",
        "resolucion",
        "filtroAplicado",
        "mascaraAplicada",
        "hilos",
        "bloques",
        "arranque"
    ].forEach(id => {
        obtener(id).textContent = "—";
    });
}


function actualizarMascaras() {

    const permitidas = filtro.value === "Media"
        ? [3, 9, 21]
        : [3];

    mascara.innerHTML = "";

    permitidas.forEach(valor => {

        const opcion = document.createElement("option");

        opcion.value = valor;
        opcion.textContent = `${valor} × ${valor}`;

        mascara.appendChild(opcion);
    });
}


async function comprobarGPU() {

    const estado = obtener("estadoGPU");

    try {

        const respuesta = await fetch("/api/salud", {
            cache: "no-store"
        });

        const datos = await respuesta.json();

        if (!respuesta.ok) {
            throw new Error(
                datos.mensaje || "GPU no disponible"
            );
        }

        estado.textContent =
            `GPU disponible: ${datos.gpu} | ` +
            `Inicialización: ${datos.inicializacion_ms.toFixed(3)} ms | ` +
            `Compilación: ${datos.compilacion_ms.toFixed(3)} ms`;

        estado.className = "correcto";

    } catch (error) {

        estado.textContent =
            `No se pudo verificar la GPU: ${error.message}`;

        estado.className = "error";
    }
}


function tiempoHeader(respuesta, nombre) {

    const valor = respuesta.headers.get(nombre);

    if (valor === null) {
        return "—";
    }

    const numero = Number(valor);

    return Number.isFinite(numero)
        ? `${numero.toFixed(3)} ms`
        : "—";
}


filtro.addEventListener("change", () => {

    actualizarMascaras();
    limpiarResultado();
});


mascara.addEventListener("change", limpiarResultado);


imagen.addEventListener("change", () => {

    limpiarResultado();

    if (urlOriginal) {
        URL.revokeObjectURL(urlOriginal);
        urlOriginal = null;
    }

    obtener("original").hidden = true;
    obtener("textoOriginal").hidden = false;

    const archivo = imagen.files[0];

    if (!archivo) {
        return;
    }

    if (archivo.size > MAX_ARCHIVO) {

        mostrarMensaje(
            "La imagen supera el límite de 40 MiB.",
            "error"
        );

        imagen.value = "";
        return;
    }

    urlOriginal = URL.createObjectURL(archivo);

    const vista = obtener("original");

    vista.onload = () => {
        vista.hidden = false;
        obtener("textoOriginal").hidden = true;
    };

    vista.onerror = () => {

        vista.hidden = true;

        obtener("textoOriginal").textContent =
            "Vista previa no compatible con el navegador. " +
            "El archivo puede enviarse al servidor.";

        obtener("textoOriginal").hidden = false;
    };

    vista.src = urlOriginal;

    mostrarMensaje("Imagen seleccionada.", "correcto");
});


formulario.addEventListener("submit", async evento => {

    evento.preventDefault();

    // Evitar solicitudes duplicadas accidentales.
    if (procesando) {
        return;
    }

    const archivo = imagen.files[0];

    if (!archivo) {

        mostrarMensaje(
            "Seleccione una imagen.",
            "error"
        );

        return;
    }

    if (archivo.size > MAX_ARCHIVO) {

        mostrarMensaje(
            "La imagen supera el límite permitido.",
            "error"
        );

        return;
    }

    limpiarResultado();

    const datos = new FormData();

    datos.append("imagen", archivo);
    datos.append("filtro", filtro.value);
    datos.append("mascara", mascara.value);
    datos.append("salida", "png");

    procesando = true;

    boton.disabled = true;
    imagen.disabled = true;
    filtro.disabled = true;
    mascara.disabled = true;

    boton.textContent = "Procesando con GPU...";

    mostrarMensaje(
        "Enviando imagen y ejecutando el kernel CUDA.",
        "procesando"
    );

    try {

        // Reloj monotónico del cliente.
        const inicio = performance.now();

        const respuesta = await fetch("/api/procesar", {
            method: "POST",
            body: datos
        });

        if (!respuesta.ok) {

            const error = await respuesta.json().catch(() => ({}));

            throw new Error(
                error.mensaje || `Error HTTP ${respuesta.status}`
            );
        }

        // Esperar la descarga completa de la imagen.
        const resultado = await respuesta.blob();

        const tiempoCliente = performance.now() - inicio;

        urlResultado = URL.createObjectURL(resultado);

        obtener("resultado").src = urlResultado;
        obtener("resultado").hidden = false;
        obtener("textoResultado").hidden = true;

        const id = respuesta.headers.get("X-Request-ID");

        const filtroUsado = respuesta.headers.get("X-Filtro");
        const mascaraUsada = respuesta.headers.get("X-Mascara");

        const descarga = obtener("descargar");

        descarga.href = urlResultado;

        descarga.download =
            `${id}_${filtroUsado}_${mascaraUsada}x${mascaraUsada}.png`;

        descarga.hidden = false;

        for (const [elemento, encabezado] of Object.entries(metricas)) {

            obtener(elemento).textContent =
                tiempoHeader(respuesta, encabezado);
        }

        obtener("cliente").textContent =
            `${tiempoCliente.toFixed(3)} ms`;

        obtener("identificador").textContent = id;

        obtener("gpu").textContent =
            respuesta.headers.get("X-gpu") || "—";

        obtener("resolucion").textContent =
            `${respuesta.headers.get("X-ancho")} × ` +
            `${respuesta.headers.get("X-alto")}`;

        obtener("filtroAplicado").textContent =
            filtroUsado;

        obtener("mascaraAplicada").textContent =
            `${mascaraUsada} × ${mascaraUsada}`;

        obtener("hilos").textContent =
            respuesta.headers.get("X-hilos");

        obtener("bloques").textContent =
            respuesta.headers.get("X-bloques");

        obtener("arranque").textContent =
            respuesta.headers.get("X-Arranque-Frio") === "1"
                ? tiempoHeader(respuesta, "X-Arranque-ms")
                : "No";

        mostrarMensaje(
            "Imagen procesada correctamente mediante PyCUDA.",
            "correcto"
        );

    } catch (error) {

        mostrarMensaje(error.message, "error");

    } finally {

        procesando = false;

        boton.disabled = false;
        imagen.disabled = false;
        filtro.disabled = false;
        mascara.disabled = false;

        boton.textContent = "Procesar imagen con GPU";
    }
});


actualizarMascaras();
comprobarGPU();
