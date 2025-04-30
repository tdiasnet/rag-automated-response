import torch

# Verifica se a GPU está disponível
if torch.cuda.is_available():
    print(f"GPU disponível: {torch.cuda.get_device_name(0)}")
else:
    print("GPU não disponível")
