import torch

torch.set_default_dtype(
    torch.float64
)  # This also implies default complex128. It applies to all scripts of the library
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # Shared with other scripts of the library
