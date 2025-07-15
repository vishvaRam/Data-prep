import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    # Get the list of CUDA architectures this library was compiled for
    supported_cuda_architectures = torch.cuda.get_arch_list()
    print(f"Supported CUDA architectures by this PyTorch build: {supported_cuda_architectures}")

    # You can also check the compute capability of your installed GPU(s)
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Compute Capability (major.minor): {torch.cuda.get_device_capability(i)}")
else:
    print("CUDA is not available. PyTorch is running on CPU.")
    print("For CPU-only builds, specific CUDA architectures are not applicable.")