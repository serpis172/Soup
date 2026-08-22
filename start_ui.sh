#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="soup-lab-fork"
CONTAINER_NAME="soup-lab-ui"

echo "🍲 Soup HomeServer Lab"
echo "Cartella di lavoro: $(pwd)"

# Rimuovi eventuale immagine precedente che potrebbe essere corrotta
if docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "🗑️  Rimozione immagine precedente..."
    docker rmi "$IMAGE_NAME" || true
fi

echo "🔨 Costruzione immagine Docker (può richiedere 10-20 minuti)..."
echo "Questo installerà e compilerà le dipendenze CUDA..."

# Build con output verbose per vedere dove fallisce se c'è un errore
if ! docker build -f Dockerfile.ui -t "$IMAGE_NAME" .; then
    echo ""
    echo "❌ Build fallita! Possibili soluzioni:"
    echo "   1. Assicurati di avere abbastanza spazio su disco (almeno 20GB)"
    echo "   2. Prova a ridurre CMAKE_BUILD_PARALLEL_LEVEL nel Dockerfile"
    echo "   3. Se fallisce solo llama-cpp-python, puoi commentarlo dal Dockerfile"
    exit 1
fi

# Rimuovi container precedente se esiste
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "🧹 Rimozione container precedente..."
    docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
fi

echo "🚀 Avvio Soup WebUI..."
echo "Apri nel browser: http://localhost:7860"
echo "ℹ️  Il token Bearer richiesto per Avviare/Fermare training e scaricare modelli"
echo "   viene stampato qui sotto all'avvio (pannello 'soup ui') — copialo se il"
echo "   browser non lo raccoglie automaticamente dall'URL."

docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size="16g" \
    -p 7860:7860 \
    -v "$(pwd)/data:/workspace/data" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd)/models:/workspace/models" \
    -v "$(pwd)/datasets:/workspace/datasets" \
    "$IMAGE_NAME"