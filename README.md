# FDS-project

---CNN---

Cose che si possono modificare manualmente (e quindi cose da provare per ottenere il miglior risultato possibile):

1) Parametri convoluzione:
   • Kernel size (dimensione del filtro (es. 3×3, 5×5))
   • Stride (di quanto si sposta il filtro a ogni passo. Stride > 1 = riduzione della risoluzione (downsampling))
   • Padding (pixel aggiunti intorno all’immagine per controllare la dimensione dell’output)
   • Numero di filtri = out_channels (ogni feature map produce un layer)
   • Tipo di convoluzione
2) Funzione di attivazione (ReLU, LeakyReLU, PReLU, GELU)
3) Normalizzazioni (Batch Normalization, Layer Normalization)
4) Pooling (max, average)
5) Inizializzazione pesi (Xavier/Glorot per tanh/sigmoid, Kaiming/He per ReLU)
6) Regolarizzazione (Dropout, Inverted Dropout, DropBlock)
7) Data augmentation
8) Struttura della rete (profondità, larghezza, skip connections)
9) Parametri di training:
    - Learning rate
    - Learning rate schedule
    - Ottimizzatore
    - Batch size
    - Early stopping
