import gradio as gr
import time
import subprocess
import os

# CSS e Tailwind per i tooltip professionali
HEAD_INJECTION = """
<script src="https://cdn.tailwindcss.com"></script>
<style>
    .info-icon { cursor: help; }
    .tooltip { visibility: hidden; opacity: 0; transition: opacity 0.2s; }
    .info-icon:hover .tooltip { visibility: visible; opacity: 1; }
</style>
"""

def tooltip(label, text):
    """Genera l'HTML per l'icona 'i' con tooltip al passaggio del mouse."""
    return f"""
    <div class="flex items-center mb-2">
        <span class="font-semibold text-gray-200">{label}</span>
        <div class="info-icon relative ml-2 text-blue-400">
            <svg xmlns="http://www.w3.org/2000/svg" class="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><circle cx="12" cy="12" r="10" stroke-width="2"/><path stroke-width="2" d="M12 16v-4m0-4h.01"/></svg>
            <div class="tooltip absolute bottom-full left-1/2 -translate-x-1/2 mb-2 bg-gray-800 text-white text-xs rounded py-1 px-3 w-64 z-50 shadow-xl border border-gray-700">
                {text}
            </div>
        </div>
    </div>
    """

def start_workflow(yaml_config, ram_cache, quant_method, progress=gr.Progress()):
    progress(0.1, desc="Scrittura configurazione temporanea...")
    
    # Crea un file yaml temporaneo con le impostazioni della UI
    with open("temp_soup.yaml", "w") as f:
        f.write(yaml_config)
        
    progress(0.3, desc=f"Avvio Soup CLI con RAM Cache {ram_cache}GB...")
    
    # Esegui il comando soup train in un processo separato
    # Questo permette a Gradio di leggere l'output in tempo reale
    process = subprocess.Popen(
        ["soup", "train", "--config", "temp_soup.yaml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd="/app" # Assumendo che il docker lavori in /app
    )
    
    output_lines = []
    # Legge l'output riga per riga e aggiorna la UI
    for line in process.stdout:
        output_lines.append(line)
        # Gradio aggiorna la textbox in tempo reale
        yield "\n".join(output_lines[-50:]) # Mostra le ultime 50 righe
        
    process.wait()
    return "✅ Workflow Completato!"

with gr.Blocks(head=HEAD_INJECTION, theme=gr.themes.Soft(primary_hue="blue"), title="Soup Lab UI") as demo:
    gr.Markdown("# 🍲 Soup - HomeServer Control Center")
    
    with gr.Tab("⚙️ Workflow & Streaming"):
        with gr.Row():
            with gr.Column():
                gr.HTML(tooltip("Configurazione YAML", "Incolla qui il contenuto del tuo soup.yaml o carica il file."))
                yaml_input = gr.Textbox(lines=10, label="")
                
                gr.HTML(tooltip("System RAM Cache (GB)", "Imposta quanta RAM usare per pre-caricare i layer successivi. Valori alti riducono i colli di bottiglia PCIe."))
                ram_slider = gr.Slider(0, 64, value=8, step=2, label="")
                
                gr.HTML(tooltip("Strategia di Quantizzazione", "Scegli il metodo di compressione. k/i-quants generano file GGUF per llama.cpp."))
                quant_dropdown = gr.Dropdown(
                    choices=["Nessuna (FP16)", "AWQ", "GPTQ", "k-quants (GGUF)", "i-quants (GGUF)", "QAT"], 
                    value="AWQ", label=""
                )
                
                run_btn = gr.Button("Avvia Workflow", variant="primary")
            
            with gr.Column():
                status_console = gr.Textbox(label="Status Real-Time", interactive=False, lines=15)
                
    run_btn.click(
        start_workflow, 
        inputs=[yaml_input, ram_slider, quant_dropdown], 
        outputs=[status_console]
    )

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, share=False)