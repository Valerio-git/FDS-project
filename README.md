# FDS-project

---CNN---

Cose che si possono modificare manualmente (e quindi cose da provare per ottenere il miglior risultato possibile):

1) Parametri convoluzione:
   - ~~Kernel size (dimensione del filtro (es. 3×3, 5×5))~~
   - ~~Stride (di quanto si sposta il filtro a ogni passo. Stride > 1 = riduzione della risoluzione (downsampling))~~
   - ~~Padding (pixel aggiunti intorno all’immagine per controllare la dimensione dell’output)~~
   - ~~Numero di filtri = out_channels (ogni feature map produce un layer)~~
2) ~~Funzione di attivazione (ReLU, LeakyReLU, PReLU, GELU)~~
3) ~~Normalizzazioni (Batch Normalization, Layer Normalization)~~
4) ~~Pooling (max, average)~~
5) ~~Inizializzazione pesi (Xavier/Glorot per tanh/sigmoid, Kaiming/He per ReLU)~~
6) ~~Regolarizzazione (Dropout, Inverted Dropout, DropBlock)~~
7) Data augmentation
8) Struttura della rete (profondità, larghezza, skip connections)
9) Parametri di training:
    - ~~Learning rate~~
    - ~~Learning rate schedule~~
    - ~~Ottimizzatore~~
    - ~~Batch size~~
    - ~~Early stopping~~

Da fare:
- Aggiungere i seed
- Switchare alle etichette generiche
- Modificare fully connected layers
- Modificare fine_tune.py per salvare modello per testing
- Aggiungere altre metriche e usarle in train_hyperparameters.py
- Fine tuning con modello preaddestrato
- Aggiungere un altro modello
- Aggiungere file *main* con pipeline che esegua tutto il codice
- Data Analysis e plotting (training/validation loss,...)
