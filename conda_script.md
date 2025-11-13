conda create -n interp_v0 python=3.10 numpy=1.26
conda activate interp_v0
conda install pytorch=2.2 pytorch-cuda=12.1 -c nvidia -c pytorch
conda install scipy pandas scikit-learn tqdm -c conda-forge
pip install einops datasets evaluate wandb matplotlib seaborn transformers transformer-lens jaxtyping gooey huggingface_hub
