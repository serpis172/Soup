import threading
import collections
import psutil

class RamLayerCache:
    def __init__(self, max_ram_gb: float):
        self.max_ram_bytes = int(max_ram_gb * (1024**3))
        self.current_size = 0
        # OrderedDict funge da cache LRU (Least Recently Used)
        self.cache = collections.OrderedDict()
        self.lock = threading.Lock()

    def get_layer(self, layer_id, load_fn):
        """Recupera un layer dalla RAM o lo carica e avvia il prefetch."""
        with self.lock:
            if layer_id in self.cache:
                self.cache.move_to_end(layer_id) # Aggiorna LRU
                return self.cache[layer_id]
        
        # Cache miss: caricamento sincrono
        layer_data = load_fn(layer_id)
        self._add_to_cache(layer_id, layer_data)
        
        # Avvia thread per pre-caricare i layer successivi
        threading.Thread(target=self._prefetch_next, args=(layer_id, load_fn), daemon=True).start()
        return layer_data

    def _prefetch_next(self, current_id, load_fn):
        """Carica i layer successivi finché c'è RAM disponibile."""
        next_id = current_id + 1
        while self.current_size < self.max_ram_bytes:
            try:
                # Simulazione o logica reale per ottenere il peso del prossimo layer
                next_data = load_fn(next_id) 
                self._add_to_cache(next_id, next_data)
                next_id += 1
            except Exception:
                break # Fine dei layer o errore

    def _add_to_cache(self, layer_id, data):
        with self.lock:
            data_size = self._get_size(data)
            while self.current_size + data_size > self.max_ram_bytes and self.cache:
                _, evicted = self.cache.popitem(last=False) # Rimuovi il più vecchio
                self.current_size -= self._get_size(evicted)
            self.cache[layer_id] = data
            self.current_size += data_size

    def _get_size(self, obj): 
        # Calcolo approssimativo della memoria dei tensori/dict
        return sum(p.numel() * p.element_size() for p in obj.values()) if hasattr(obj, 'values') else 0