FROM node:22-bookworm-slim AS assets
WORKDIR /build
COPY docker/assets/package.json docker/assets/package-lock.json docker/assets/tailwind.config.cjs docker/assets/input.css ./
RUN npm ci --no-audit --no-fund
COPY web/static ./web/static
RUN npx tailwindcss -c tailwind.config.cjs -i input.css -o output/tailwind.css --minify \
    && mkdir -p output/fontawesome/css output/fontawesome/webfonts \
    && cp node_modules/@fortawesome/fontawesome-free/css/all.min.css output/fontawesome/css/all.min.css \
    && cp node_modules/@fortawesome/fontawesome-free/webfonts/* output/fontawesome/webfonts/ \
    && cp node_modules/@fortawesome/fontawesome-free/LICENSE.txt output/FONT_AWESOME_LICENSE.txt

FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 YOLO_CONFIG_DIR=/app/local_only \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 libgomp1 fonts-thai-tlwg \
    && rm -rf /var/lib/apt/lists/*
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
# Keep the large CPU/CUDA installation independent of app/dependency revisions.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch==2.11.0 torchvision==0.26.0 --index-url ${TORCH_INDEX}
COPY docker/requirements.lock.txt /tmp/requirements.lock.txt
COPY docker/torch.constraints.txt /tmp/torch.constraints.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --only-binary=:all: -c /tmp/torch.constraints.txt -r /tmp/requirements.lock.txt \
    && pip check
COPY app ./app
COPY integration ./integration
COPY web ./web
COPY Fall/Project.py ./Fall/Project.py
COPY Location/location ./Location/location
COPY Wandering/ai_camera_system ./Wandering/ai_camera_system
COPY docker/entrypoint.py docker/healthcheck.py ./docker/
COPY --from=assets /build/output ./web/static/vendor
RUN mkdir -p /app/local_only /app/models /incoming_cctv
ARG WEB_REVISION=qa-fall-lstm-main-20261007-fall54fix1
ARG NEXORA_VERSION=v0.1.0-rc.1
LABEL org.opencontainers.image.title="NEXORA" org.opencontainers.image.version="${NEXORA_VERSION}" nexora.web.revision="${WEB_REVISION}"
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "/app/docker/healthcheck.py"]
ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD ["python", "-m", "uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8080"]
