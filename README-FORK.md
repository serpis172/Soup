# 🍲 Soup — questo fork

Fork di [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup): tutti i meriti per
il design originale, il layer streaming e il CLI di base vanno al progetto upstream. Questo
documento elenca **solo** cosa cambia in questo fork e come si intreccia con upstream — per
tutto il resto (comandi base, formati dati, task supportati) vale la documentazione
principale in [`README.md`](README.md) e in [`docs/`](docs/).

> Nota di accuratezza: una versione precedente di questo file descriveva una WebUI basata su
> Gradio. Non è più così — la WebUI di questo fork è un'app FastAPI + HTML/JS statico
> (`soup_cli/ui/app.py` + `soup_cli/ui/static/`), senza dipendenza da Gradio. Il testo sotto
> riflette lo stato attuale del codice, non quello di quando questo file è stato scritto la
> prima volta.

## Novità di questo fork

### WebUI (FastAPI, non Gradio)
- Avvio con `soup ui` — apre `http://127.0.0.1:<porta>/?token=...`; il token in query string
  è obbligatorio: se apri l'URL senza, ogni chiamata fallisce con `401 Unauthorized`. La UI
  ora mostra un banner esplicito quando manca il token, con un campo per incollarlo, invece
  di lasciare che ogni pulsante fallisca silenziosamente.
- Pannelli separati per **RAM Prefetch** e **Quantizzazione** (`/api/config/patch-ram-prefetch`
  e `/api/config/patch-quant`), invece di un unico pannello/endpoint combinato.
- Scan di importanza (magnitude / Wanda) con calibrazione multi-dataset: incolla testo e/o
  elenca uno o più file JSONL di calibrazione, pool campionato da ciascuno.

### RAM prefetch predittivo per layer streaming
- Thread in background che pre-carica in RAM pinned i prossimi layer del decoder, per
  checkpoint troppo grandi per stare interamente in VRAM/RAM.
- Configurabile via YAML con `training.ram_cache_gb` (separato dalla quantizzazione — vedi
  sopra).

### Quantizzazione: più opzioni di bit-width
- **AWQ** — solo 4-bit (limite della libreria `autoawq`, non di Soup: l'algoritmo è
  4-bit-only in ogni implementazione mainstream). Prima di questo fork era possibile
  richiedere 8-bit e fallire tardi, a export in corso; ora viene rifiutato subito con un
  messaggio che spiega perché.
- **GPTQ** — **2 / 3 / 4 / 8-bit** (prima limitato a 4/8 in tre punti diversi del codice,
  nonostante `quant_menu.build_gptq_config` supportasse già 2/3-bit).
- **k-quants / i-quants** (formati GGUF, per llama.cpp).
- Vedi `training.custom_quant_strategy` + `training.custom_quant_detail` nello schema di
  configurazione, e `soup export --bits`.

### Pipeline di compressione opzionale nel training YAML
`training.pipeline` — tre stage opzionali, ordine fisso, tutti disattivati di default (zero
impatto su config esistenti):

1. **`activation_scan`** — scan di importanza (magnitude o Wanda) sul checkpoint, prima di
   qualsiasi compressione. Con Wanda, la calibrazione arriva da `data.calibration` (uno o più
   file) o, in fallback, da `data.train`.
2. **`compress`** — merge di neuroni simili (soglia coseno) o compressione SVD, scrive un
   nuovo checkpoint locale.
3. **`distill`** — marcatore di posizione: la distillazione vera e propria resta
   `task: distill` con i campi `training.distill_*` / `training.teacher_model` già esistenti
   in upstream; questo flag richiede solo che `task='distill'` sia coerente col resto.

Eseguita esplicitamente con `soup pipeline run config.yaml` (non da `soup train`
automaticamente — vedi il docstring di `soup_cli/utils/pipeline_orchestrator.py` per il
perché: le funzioni di merge/SVD sottostanti non avevano copertura di test in upstream prima
di questo fork, quindi riscrivere un checkpoint come effetto collaterale silenzioso di
`soup train` è stato deliberatamente evitato).

### Training multi-obiettivo (`training.objectives`)
Dichiara uno o più domini SFT nella stessa run (`code`, `tool_call`, `reasoning`, `chat`,
`general` — liberamente combinabili sotto `task: sft`/`distill`), oppure `orpo` da solo
(richiede `task: orpo`, non combinabile con gli altri: ORPO addestra su triplette
prompt/chosen/rejected, non sulle righe a singola risposta che gli obiettivi SFT assumono).

### Dataset multipli per train / val / calibrazione
- `data.train`, `data.val` (nuovo — prima la validazione poteva solo derivare da
  `data.train` via `val_split`) e `data.calibration` (nuovo) accettano un singolo percorso
  o una lista, concatenati nell'ordine dato.
- `data.verify_before_training` (default `true`): prima che il training parta, ogni sorgente
  locale viene verificata con gli stessi controlli di `soup data inspect` (righe utilizzabili,
  duplicati, file mancanti) — fallisce subito invece di scoprire un dataset vuoto dopo aver
  allocato modello e ottimizzatore.

### Docker
- `Dockerfile.ui` dedicato, entrypoint `soup ui`, senza dipendenza Gradio.

## Requisiti (WebUI)
- GPU NVIDIA con VRAM sufficiente per il modello scelto (il layer streaming di upstream
  riduce il requisito, ma non lo azzera).
- Per `--metric wanda` (scan attivazioni): il modello deve essere caricabile per intero
  almeno una volta.

## Cosa NON è cambiato
Task supportati, formati dati, comandi CLI di base (`soup train`, `soup export`,
`soup data inspect`, ecc.), formato dei checkpoint prodotti da un training "semplice" (senza
`training.pipeline`) — tutto identico a upstream. Questo fork aggiunge funzionalità opt-in,
non ne ridefinisce di esistenti.

## Quick start (WebUI via Docker)

Prerequisiti: Docker con NVIDIA Container Toolkit, GPU con VRAM sufficiente per il modello
scelto, RAM sufficiente per `training.ram_cache_gb` se lo usi.

```bash
git clone https://github.com/serpis172/Soup.git
cd Soup
chmod +x start_ui.sh
./start_ui.sh              # build (cache-aware) + avvio
./start_ui.sh --rebuild    # forza un'immagine pulita
./start_ui.sh --skip-build # riavvia l'immagine esistente senza rebuild
```

Lo script stampa l'URL con `?token=...` da aprire — vedi la nota sul banner "Session token
needed" più sopra se apri l'URL senza il token.
