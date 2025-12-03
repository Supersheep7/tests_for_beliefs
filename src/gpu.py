import runpod

def setup_gpu():
    
    client = runpod.Client(api_key="YOUR_API_KEY")
    pod = client.pods.create(
        name="my-a100",
        gpu_type="A100",
        cloud_type="SECURE",
        container="runpod/pytorch:latest",
        gpu_count=1
    )

    import torch
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, using CPU.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device