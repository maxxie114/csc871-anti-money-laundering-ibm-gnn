# save as test_cuda.py and run: python3 test_cuda.py

import platform

print("=== Environment ===")
print("Platform:", platform.platform())
print("Python:", platform.python_version())

try:
    import torch
except ImportError as e:
    print("\nPyTorch is not installed or not in this Python environment.")
    raise SystemExit(e)

print("\n=== PyTorch / CUDA Info ===")
print("torch.__version__:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)

cuda_available = torch.cuda.is_available()
print("torch.cuda.is_available():", cuda_available)

if not cuda_available:
    print("\nCUDA is NOT available to PyTorch in this environment.")
else:
    # Number of devices
    device_count = torch.cuda.device_count()
    print("torch.cuda.device_count():", device_count)

    for i in range(device_count):
        print(f"  device {i}: {torch.cuda.get_device_name(i)}")

    # Simple tensor test on GPU
    try:
        x = torch.rand(3, 3, device="cuda")
        y = torch.rand(3, 3, device="cuda")
        z = x @ y
        print("\nSuccessfully ran a matrix multiply on CUDA.")
        print("z.device:", z.device)
    except Exception as e:
        print("\nERROR: Allocation or compute on CUDA failed:")
        print(e)

