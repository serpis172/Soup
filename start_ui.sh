#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

IMAGE_NAME="soup-lab-fork"
CONTAINER_NAME="soup-lab-ui"

# Bug fix: this script used to `docker rmi` the previous image before every
# single run, unconditionally, on the theory it "might be corrupted" — that
# throws away a perfectly good image (and, combined with the Dockerfile
# previously COPYing full source before the expensive pip/CUDA-compile
# steps, forced a full from-scratch reinstall of torch/llama-cpp-python
# every launch). Dockerfile.ui is now ordered so dependency layers only
# rebuild when pyproject.toml changes; this script now just lets `docker
# build` do its normal cache-hit-or-miss thing, and BuildKit's pip cache
# mount (--mount=type=cache in the Dockerfile) speeds up even genuine
# dependency-layer rebuilds. Use --rebuild to force a clean image if you
# really do suspect corruption, and --skip-build to relaunch the existing
# image with no `docker build` call at all (useful if you only changed
# something outside the image, like your data/ folder).
export DOCKER_BUILDKIT=1

REBUILD=0
SKIP_BUILD=0
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=1 ;;
        --skip-build) SKIP_BUILD=1 ;;
        -h|--help)
            echo "Usage: $0 [--rebuild] [--skip-build]"
            echo "  --rebuild     Force a clean image (docker rmi first, then full build)"
            echo "  --skip-build  Skip 'docker build' entirely, just (re)start the existing image"
            exit 0
            ;;
    esac
done

echo "🍲 Soup HomeServer Lab"
echo "Cartella di lavoro: $(pwd)"

if [ "$REBUILD" -eq 1 ] && docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo "🗑️  --rebuild richiesto: rimozione immagine precedente..."
    docker rmi "$IMAGE_NAME" || true
fi

if [ "$SKIP_BUILD" -eq 1 ]; then
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        echo "❌ --skip-build richiesto ma nessuna immagine '$IMAGE_NAME' esiste ancora."
        echo "   Rilancia senza --skip-build per costruirla la prima volta."
        exit 1
    fi
    echo "⏭️  --skip-build: uso l'immagine esistente senza ricostruire."
else
    echo "🔨 Costruzione immagine Docker..."
    echo "   Prima volta: 10-20 minuti (compila CUDA/llama.cpp)."
    echo "   Volte successive: solo i layer che sono davvero cambiati —"
    echo "   se non hai toccato pyproject.toml, questo passo dura secondi."

    if ! docker build -f Dockerfile.ui -t "$IMAGE_NAME" .; then
        echo ""
        echo "❌ Build fallita! Possibili soluzioni:"
        echo "   1. Assicurati di avere abbastanza spazio su disco (almeno 20GB)"
        echo "   2. Prova a ridurre CMAKE_BUILD_PARALLEL_LEVEL nel Dockerfile"
        echo "   3. Se fallisce solo llama-cpp-python, puoi commentarlo dal Dockerfile"
        echo "   4. Immagine sospetta corrotta? Rilancia con --rebuild"
        exit 1
    fi
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

mkdir -p data output models datasets

docker run -it --rm \
    --name "$CONTAINER_NAME" \
    --gpus all \
    --shm-size="16g" \
    -p 7860:7860 \
    -v "$(pwd)/data:/workspace/data" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd)/models:/workspace/models" \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "soup-hf-cache:/root/.cache/huggingface" \
    "$IMAGE_NAME"
