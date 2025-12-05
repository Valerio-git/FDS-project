from src.models.CNN import CNN
import torch

def load_cnn_from_checkpoint(checkpoint_path: str, num_classes: int, 
                             device: torch.device | None = None, strict: bool = True) -> CNN:
    model = CNN(num_classes = num_classes)
    state_dict = torch.load(checkpoint_path, map_location = "cpu")
    model.load_state_dict(state_dict, strict = strict)

    if device is not None:
        model.to(device)

    return model


def freeze_all_except_classifier(model: CNN) -> None:
    """
    STEP 1 del fine-tuning:
    - congela tutti i layer convoluzionali
    - allena solo il classifier (fc1 e fc2)

    Da usare per le prime poche epoche su dataset White.
    """
    for name, param in model.named_parameters():
        if name.startswith("fc1") or name.startswith("fc2"):
            param.requires_grad = True
        else:
            param.requires_grad = False
    '''model.named_parameters() contiene tutti i parametri del modello che a seconda che siano
    di un conv layer o di un fully connected verranno chiamati: conv1.weight, conv2.bias,...,
    fc1.weigth, fc2bias...
    poi calcola il gradiente (quindi allena) solo fc1 e fc2'''


def unfreeze_last_conv_block(model: CNN) -> None:
    """
    STEP 2 del fine-tuning:
    - rende allenabili conv3, bn3, fc1 e fc2
    - conv1/bn1/conv2/bn2 restano congelati (feature più generiche)

    Da usare dopo lo STEP 1, per un adattamento un po' più profondo.
    """
    for name, param in model.named_parameters():
        if name.startswith(("conv3", "bn3", "fc1", "fc2")):
            param.requires_grad = True
        # qui congela solamente i primi due layer convoluzionali


def get_trainable_parameters(model: CNN):
    """
    Tramite model.named_parameters ho estratto tutti i parametri, ognuno ha un attributo
    param.requires_grad che se = True permette a pyhton di calcolare il gradiente.
    Qui filtriamo solamente i parametri = True (quindi no quelli dei layer congelati)
    """
    return filter(lambda p: p.requires_grad, model.parameters())