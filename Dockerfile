
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

LABEL maintainer="Proyecto Computacion Paralela UPS"
LABEL description="Servicio web de convolucion con Docker y PyCUDA"
LABEL version="1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CUDA_HOME=/usr/local/cuda
ENV CUDA_ROOT=/usr/local/cuda
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Instalar Python y herramientas de compilacion.
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Directorio principal de la aplicacion.
WORKDIR /app

# Instalar dependencias de Python.
COPY requirements.txt .

RUN python3 -m pip install --no-cache-dir \
    pip==25.0.1 \
    setuptools==75.8.0 \
    wheel==0.45.1

RUN python3 -m pip install --no-cache-dir \
    -r requirements.txt

# Copiar archivos de la aplicacion.
COPY . .

# Crear directorios para resultados.
RUN mkdir -p resultados graficas

# Puerto del servicio web.
EXPOSE 8000

# Servidor de produccion.
# Un proceso y cuatro hilos HTTP.
# El acceso a la GPU se controla desde app.py.
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "1", \
     "--threads", "4", \
     "--timeout", "300", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "app:app"]
