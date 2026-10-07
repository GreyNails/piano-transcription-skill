FROM node:22-bookworm-slim AS node
FROM python:3.11-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm \
    && apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg librsvg2-bin fonts-dejavu-core build-essential \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
ARG TORCH_VARIANT=cu121
COPY requirements.txt requirements.lock package*.json ./
RUN python -m pip install --no-cache-dir torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/${TORCH_VARIANT} \
    && python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x deploy.sh run.sh && ./deploy.sh --python /usr/local/bin/python
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["/app/run.sh"]
