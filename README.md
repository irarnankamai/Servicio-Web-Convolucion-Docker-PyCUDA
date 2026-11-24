\# Desarrollo y despliegue de un servicio web de convolución de imágenes con Docker y PyCUDA



\*\*UNIVERSIDAD POLITÉCNICA SALESIANA (UPS)\*\*



\- Asignatura: Computación Paralela

\- Práctica de laboratorio: 03

\- Docente: Gabriel A. León Paredes

\- Estudiante: Noe Ayavaca - Irar Nankamai

\- Fecha: 24 de noviembre de 2026



\## 1. Descripción del proyecto



Este proyecto implementa un servicio web para el procesamiento paralelo de imágenes de alta resolución mediante convolución en GPU, utilizando Python, PyCUDA, Flask, Gunicorn y Docker.



La aplicación permite cargar imágenes, seleccionar filtros y tamaños de máscara, ejecutar el procesamiento en una GPU NVIDIA, visualizar y descargar los resultados, y consultar los tiempos de ejecución.



\## 2. Continuidad de la Práctica 02



El proyecto reutiliza el kernel CUDA y los filtros desarrollados en la Práctica 02.



Repositorio original:



https://github.com/irarnankamai/Convoluci-n-de-im-genes-de-alta-resoluci-n-mediante-PyCUDA.



Commit de referencia:



`6bcedf33b41254a4dcf8f22fba123caa06782d13`



Se mantienen los coeficientes de las máscaras, la precisión float32, la extensión de ceros en los bordes y la asignación de un píxel de salida por hilo.



Las principales modificaciones corresponden a la administración explícita del contexto CUDA, la integración con el servicio HTTP, el manejo de memoria, la validación de solicitudes y el control de concurrencia.



\## 3. Tecnologías utilizadas



\- Python

\- PyCUDA

\- CUDA

\- NumPy

\- Pillow

\- Flask

\- Gunicorn

\- Docker Desktop

\- HTML, CSS y JavaScript

\- Matplotlib

\- Git y GitHub



\## 4. Arquitectura



```text

Cliente web

&#x20;    |

&#x20;    | HTTP POST

&#x20;    v

Flask / Gunicorn

&#x20;    |

Validacion de solicitudes

&#x20;    |

Control de capacidad y bloqueo

&#x20;    |

Motor PyCUDA

&#x20;    |

GPU NVIDIA

&#x20;    |

Resultado float32

&#x20;    |

Codificacion PNG / NPY / ZIP

&#x20;    |

Respuesta HTTP

```



La aplicación utiliza un proceso Gunicorn con cuatro hilos y admite hasta dos solicitudes de procesamiento. El acceso al motor GPU se encuentra serializado mediante un bloqueo.



Cuando la capacidad está ocupada, las solicitudes adicionales reciben una respuesta HTTP 429.



\## 5. Entorno experimental



| Componente | Especificación |

|---|---|

| Equipo | Dell OptiPlex 7000 |

| Procesador | Intel Core i9-12900K |

| Núcleos | 16 |

| Procesadores lógicos | 24 |

| Memoria RAM | 31.7 GB |

| GPU | NVIDIA GeForce RTX 3070 |

| Memoria GPU | 8 GB |

| Sistema anfitrión | Windows 11 Pro |

| Driver NVIDIA anfitrión | 610.60 |

| CUDA anfitrión | 12.9 |

| Docker Desktop | 4.48.0 |

| Docker Engine | 28.5.1 |

| Backend | WSL2 |

| Imagen base | nvidia/cuda:12.8.1-devel-ubuntu22.04 |

| Python en Docker | 3.10.12 |

| PyCUDA en Docker | 2025.1.2 |

| NumPy en Docker | 1.26.4 |

| Pillow en Docker | 11.1.0 |

| Flask | 3.1.0 |

| Gunicorn | 23.0.0 |

| Puerto | 8000 |



El despliegue experimental se realizó localmente utilizando Docker Desktop y una GPU NVIDIA física.



\## 6. Requisitos



\- GPU NVIDIA compatible con CUDA.

\- Controladores NVIDIA instalados.

\- Docker con soporte para GPU.

\- Git.

\- Conexión a Internet para construir la imagen Docker.



\## 7. Construcción de la imagen Docker



Desde la raíz del proyecto:



```bash

docker build -t practica3-pycuda:1.1 .

```



Verificar el acceso a la GPU:



```bash

docker run --rm --gpus all practica3-pycuda:1.1 nvidia-smi

```



Verificar CUDA:



```bash

docker run --rm --gpus all practica3-pycuda:1.1 nvcc --version

```



\## 8. Ejecución del servicio



```bash

docker run -d --name practica3-web --gpus all -p 8000:8000 practica3-pycuda:1.1

```



Acceder desde el navegador:



http://localhost:8000



Verificar el contenedor:



```bash

docker ps

```



Consultar el estado del servicio:



```bash

curl http://localhost:8000/api/salud

```



\## 9. API REST



\### Procesamiento de imágenes



\- Método: POST

\- Ruta: `/api/procesar`

\- Formato de solicitud: multipart/form-data



| Parámetro | Descripción |

|---|---|

| imagen | Archivo de imagen |

| filtro | Media, Sharpen o Laplaciano |

| mascara | Tamaño de máscara |

| salida | png, npy o zip |



\### Filtros disponibles



| Filtro | Máscaras |

|---|---|

| Media | 3×3, 9×9 y 21×21 |

| Sharpen | 3×3 |

| Laplaciano | 3×3 |



Ejemplo de solicitud:



```bash

curl -X POST http://localhost:8000/api/procesar -F "imagen=@imagenes/imagen\_2000.pgm" -F "filtro=Media" -F "mascara=9" -F "salida=png" -o resultado.png

```



\### Códigos de respuesta



| Código | Significado |

|---|---|

| 200 | Procesamiento correcto |

| 400 | Solicitud o parámetros inválidos |

| 413 | Archivo demasiado grande |

| 429 | Capacidad del servicio ocupada |

| 503 | Servicio o GPU no disponible |



\### Límites



\- Tamaño máximo de archivo: 40 MiB.

\- Dimensiones máximas: 6000×6000 píxeles.

\- Máximo de solicitudes admitidas simultáneamente: 2.

\- Hilos CUDA por bloque: 256.



\## 10. Validación numérica



Se utilizaron matrices pequeñas, imágenes de dimensiones irregulares y comparaciones entre ejecución directa y servicio HTTP.



La tolerancia de error absoluto establecida fue de 0.0001.



| Validación | Resultado |

|---|---|

| Matriz manual 5×5 | Correcta |

| CPU frente a GPU | Correcta |

| Dimensiones irregulares | Correcta |

| Ejecución directa frente a HTTP | Correcta |

| Total | 16/16 |

| Error máximo observado | 0.0 |

| Error medio observado | 0.0 |



La comparación numérica se realizó utilizando resultados float32 antes del recorte y la codificación de imágenes.



Ejecutar las pruebas:



```bash

docker exec practica3-web python3 -m pruebas.validar\_numerica

docker exec practica3-web python3 -m pruebas.validar\_servicio

```



\## 11. Validación del servicio



Se realizaron 13 solicitudes de validación, todas con el comportamiento esperado.



Se comprobó el manejo de archivos inválidos, archivos vacíos, filtros desconocidos, máscaras incorrectas, límites de tamaño y dimensiones, aislamiento de resultados y recuperación del servicio.



Las pruebas adicionales de concurrencia confirmaron el rechazo HTTP 429 cuando se superó la capacidad disponible.



\## 12. Metodología experimental



Se realizaron cuatro experimentos:



\- E1: comparación entre Windows directo, Docker directo y servicio web.

\- E2: variación de resolución.

\- E3: variación de filtros y máscaras.

\- E4: concurrencia de uno y dos clientes.



Se utilizaron cinco repeticiones por configuración después del calentamiento. Los resultados se analizaron mediante la mediana y la desviación estándar muestral.



Se registraron los tiempos de preparación, reserva de memoria, transferencia H2D, kernel, transferencia D2H, procesamiento GPU, codificación de salida, servidor, cliente y espera.



Los tiempos del kernel se midieron mediante eventos CUDA sincronizados. Los tiempos generales se midieron mediante un reloj monotónico de alta resolución.



\## 13. Resultados experimentales



\### E1. Modalidades de ejecución



| Modalidad | Kernel (ms) | Tiempo total (ms) |

|---|---:|---:|

| Windows directo | 2.202 | 409.856 |

| Docker directo | 2.193 | 702.703 |

| Servicio web | 12.616 | 859.593 |



El tiempo total del servicio web corresponde al tiempo percibido por el cliente.



!\[Comparación de modalidades](graficas/02\_total\_modalidades.png)



\### E2. Resoluciones



| Resolución | Kernel (ms) | Servidor (ms) | Cliente (ms) |

|---|---:|---:|---:|

| 2000×2000 | 2.568 | 244.362 | 248.510 |

| 4000×4000 | 13.421 | 821.635 | 846.058 |

| 6000×6000 | 17.397 | 1547.750 | 1579.483 |



!\[Resoluciones](graficas/05\_total\_resolucion.png)



\### E3. Filtros y máscaras



| Filtro | Máscara | Kernel (ms) | Cliente (ms) |

|---|---|---:|---:|

| Media | 3×3 | 8.601 | 1088.217 |

| Media | 9×9 | 10.021 | 815.390 |

| Media | 21×21 | 45.072 | 674.778 |

| Sharpen | 3×3 | 8.619 | 1141.793 |

| Laplaciano | 3×3 | 8.705 | 1304.939 |



!\[Filtros](graficas/06\_kernel\_filtros.png)



\### E4. Concurrencia



| Indicador | Un cliente | Dos clientes |

|---|---:|---:|

| Latencia mediana (ms) | 766.444 | 1350.004 |

| Espera mediana (ms) | 0.00038 | 396.551 |

| Throughput (req/s) | 1.302 | 1.127 |

| Solicitudes correctas | 5/5 | 10/10 |



!\[Concurrencia](graficas/09\_latencia\_concurrencia.png)



\## 14. Comparación visual



!\[Comparación visual de filtros](graficas/13\_comparacion\_visual.png)



Las imágenes se comparan utilizando la misma región espacial de 512×512 píxeles.



\## 15. Reproducción de los experimentos



Ejecución directa en Windows, con Python, PyCUDA, CUDA, compilador compatible, NumPy y Pillow previamente instalados:



```bash

python -m experimentos.ejecucion\_directa --modo windows

```



Ejecución directa en Docker:



```bash

docker exec practica3-web python3 -m experimentos.ejecucion\_directa --modo docker\_directo

```



Experimentos web:



```bash

docker exec practica3-web python3 -m experimentos.experimentos\_web

```



Concurrencia:



```bash

docker exec practica3-web python3 -m experimentos.concurrencia

```



Copiar resultados:



```bash

docker cp practica3-web:/app/resultados/. ./resultados/

```



Generar gráficas desde el anfitrión:



```bash

python -m experimentos.generar\_graficas

```



Para generar las gráficas se requiere NumPy, Pillow y Matplotlib.



\## 16. Resultados generales



\- 16 validaciones numéricas correctas.

\- 13 validaciones del servicio correctas.

\- 70 mediciones experimentales correctas.

\- 15 gráficas generadas.

\- Procesamiento de imágenes de hasta 36 millones de píxeles.

\- Aislamiento concurrente comprobado.

\- Rechazo controlado de solicitudes adicionales.

\- Recuperación del servicio después de la sobrecarga.



\## 17. Limitaciones



Las pruebas se realizaron en un entorno local con Docker Desktop y WSL2. Las versiones de las dependencias entre Windows y Docker no son idénticas, por lo que las diferencias observadas no pueden atribuirse exclusivamente a la contenerización.



La ejecución serializada protege el acceso al motor GPU, pero incrementa la latencia cuando existen solicitudes concurrentes.



Se observó variabilidad en los tiempos del kernel, especialmente durante las pruebas de concurrencia.



Los clientes de las pruebas web automatizadas se ejecutaron dentro del contenedor, mediante localhost. Por tanto, los resultados no representan la latencia de una red externa.



\## 18. Conclusiones



La integración de PyCUDA con Flask y Docker permitió implementar un servicio web funcional para el procesamiento paralelo de imágenes de alta resolución.



La validación numérica confirmó que los resultados obtenidos mediante el servicio conservan la equivalencia con la ejecución directa en las configuraciones evaluadas.



Los resultados experimentales demostraron que el rendimiento total depende tanto del procesamiento GPU como de la preparación de datos, las transferencias de memoria y la codificación de imágenes.



\## 19. Detener el servicio



```bash

docker stop practica3-web

docker rm practica3-web

```



\## 20. Autoría



Proyecto académico desarrollado para la asignatura de Computación Paralela de la Universidad Politécnica Salesiana.

