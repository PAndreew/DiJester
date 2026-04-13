FROM python:3.11-slim

# PortAudio (sounddevice) + PulseAudio client
RUN apt-get update && apt-get install -y --no-install-recommends \
        pulseaudio-utils curl libsndfile1 gosu \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/sh appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV HOME=/home/appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

ENTRYPOINT ["/entrypoint.sh"]
