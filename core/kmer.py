import numpy as np

class KmerEmbedder:
    """
    Estrae un embedding basato sulle frequenze dei k-meri direttamente
    dalla rappresentazione CGR (Chaos Game Representation).
    
    In una CGR, la griglia rappresenta spazialmente i k-meri.
    Una griglia di risoluzione R x R contiene implicitamente informazioni
    sui k-meri dove k = log2(R).
    
    Se chiediamo un k inferiore alla risoluzione della griglia, eseguiamo
    un 'pooling' (somma) per aggregare i blocchi.
    """
    
    def __init__(self, k_size: int = 3, normalize: bool = True):
        """
        Args:
            k_size (int): La lunghezza del k-mero (es. 3 per triplette).
                          Dimensione output sarà 4^k.
            normalize (bool): Se True, restituisce frequenze (somma=1).
                              Se False, restituisce conteggi grezzi.
        """
        self.k = k_size
        self.normalize = normalize
        # Dimensione target della griglia laterale (es. k=3 -> 2^3 = 8x8)
        self.target_dim = 2 ** k_size
        self.feature_dim = 4 ** k_size

    def compute_embedding(self, cgr_grid: np.ndarray) -> np.ndarray:
        """
        Converte una griglia CGR in un vettore di frequenze k-mer.
        
        Args:
            cgr_grid (np.ndarray): Matrice quadrata (NxN) della CGR.
            
        Returns:
            np.ndarray: Vettore 1D di lunghezza 4^k.
        """
        H, W = cgr_grid.shape
        
        # 1. Controllo di compatibilità
        # La risoluzione della griglia deve essere >= alla risoluzione richiesta da k
        if H < self.target_dim or W < self.target_dim:
            raise ValueError(
                f"La griglia CGR ({H}x{W}) è troppo piccola per estrarre k-meri di lunghezza {self.k}. "
                f"Serve almeno una risoluzione {self.target_dim}x{self.target_dim}."
            )

        # 2. Se la griglia è già della dimensione giusta, appiattisci
        if H == self.target_dim and W == self.target_dim:
            counts = cgr_grid.flatten()
            
        # 3. Se la griglia è più grande, facciamo Downsampling (Sum Pooling)
        else:
            # Calcoliamo quanto è grande il blocco da sommare
            # Es. Griglia 128x128, k=3 (Target 8x8). Block = 128/8 = 16.
            block_h = H // self.target_dim
            block_w = W // self.target_dim
            
            # Magia NumPy per fare il pooling efficiente senza cicli for
            # Rimodelliamo in (Target_H, Block_H, Target_W, Block_W)
            reshaped = cgr_grid.reshape(self.target_dim, block_h, self.target_dim, block_w)
            
            # Sommiamo sugli assi dei blocchi (axis 1 e 3)
            downsampled = reshaped.sum(axis=(1, 3))
            
            counts = downsampled.flatten()

        # 4. Normalizzazione (Frequenze vs Conteggi)
        if self.normalize:
            total = counts.sum()
            if total > 0:
                return counts / total
            return counts # Ritorna zeri se vuoto
        
        return counts

    def get_feature_info(self):
        return {
            "type": "k-mer frequencies",
            "k": self.k,
            "dimensions": self.feature_dim
        }