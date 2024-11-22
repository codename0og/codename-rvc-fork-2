import torch

def check_bfloat16_support():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Please check if you have a compatible GPU.")
        return False

    # Get the current GPU device
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"Checking bfloat16 support on GPU: {gpu_name}")

    # Try to create a bfloat16 tensor on the GPU
    try:
        tensor = torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        print("bfloat16 is supported on this GPU.")
        return True
    except RuntimeError as e:
        print("bfloat16 is not supported on this GPU.")
        print(f"Error: {e}")
        return False

# Run the check
bfloat16_supported = check_bfloat16_support()
