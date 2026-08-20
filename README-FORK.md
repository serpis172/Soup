# 🍲 Soup HomeServer Fork

Fork di [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup) ottimizzato per HomeServer e Lab con:

## 🆕 Nuove Funzionalità

### 🖥️ WebUI Professionale
- Interfaccia Gradio 5 con design professionale
- Tooltip informativi (icona "i") per ogni funzionalità
- Monitoraggio workflow in tempo reale
- Avvio con un click tramite `start_ui.sh`

### 💾 RAM Caching Predittivo
- Cache LRU (Least Recently Used) per i layer del modello
- Prefetching in background dei layer successivi
- Configurabile via YAML con `training.ram_cache_gb`
- Riduce i colli di bottiglia PCIe durante lo streaming

### 📊 Strategie di Quantizzazione Avanzate
- **AWQ** (Activation-aware Weight Quantization)
- **GPTQ** (4-bit quantization)
- **k-quants / i-quants** (GGUF format per llama.cpp)
- **QAT** (Quantization-Aware Training)

### 🐳 Docker Optimizzato
- Dockerfile.ui dedicato con supporto CUDA 12.1
- Installazione automatica delle dipendenze
- Supporto GPU tramite NVIDIA Container Toolkit

## 🚀 Quick Start

### Prerequisiti
- Docker con supporto GPU (NVIDIA Container Toolkit)
- Almeno 16GB RAM
- GPU NVIDIA con almeno 4GB VRAM

### Installazione

```bash
# Clona la fork
git clone https://github.com/TUO-USERNAME/soup-homeserver.git
cd soup-homeserver

# Rendi eseguibile lo script di avvio
chmod +x start_ui.sh

# Avvia (la prima volta compilerà l'immagine Docker)
./start_ui.sh