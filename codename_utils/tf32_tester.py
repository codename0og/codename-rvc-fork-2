import torch

# Ensure TensorFloat-32 (TF32) is enabled
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Confirm TF32 is enabled (should output True)
print(torch.backends.cuda.matmul.allow_tf32)
print(torch.backends.cudnn.allow_tf32)
