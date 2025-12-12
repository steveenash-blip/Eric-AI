# Install & Run (meta-llama/Llama-2-7b-chat-hf) — GPU (bitsandbytes 4-bit)

Prereqs
- NVIDIA driver + CUDA 11.8 installed and working (verify with nvidia-smi).
- conda (Miniconda/Anaconda).

Steps

1) Create and activate conda env
   conda create -n localai python=3.10 -y
   conda activate localai

2) Install PyTorch with CUDA 11.8
   conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia

3) Install Python packages
   pip install -U transformers accelerate bitsandbytes huggingface_hub sentence-transformers faiss-cpu jinja2 tqdm python-dotenv

4) Login to Hugging Face and accept Llama 2 license
   huggingface-cli login
   # After logging in, go to https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and accept the license if required.

5) Download the model (recommended to snapshot locally)
   python download_model.py --repo-id meta-llama/Llama-2-7b-chat-hf --local-dir models/llama-2-7b-chat-hf

   (If you prefer not to pre-download, run run_gpu_llama2.py with --model-id meta-llama/Llama-2-7b-chat-hf and the transformers cache will download automatically, but explicit snapshot is recommended.)

6) Generate a manifest (example)
   python run_gpu_llama2.py --model-dir models/llama-2-7b-chat-hf --prompt "Create a minimal Flutter todo app scaffold for small businesses, with offline sync."

7) Write scaffold files (use scaffold_writer.py in repo)
   python scaffold_writer.py --manifest manifest.json --out-dir generated_app

Troubleshooting
- OOM on GPU: lower --max-new-tokens, or use CPU offload with accelerate (I can provide an accelerate config if needed).
- bitsandbytes errors: ensure CUDA, GCC versions are compatible; reinstall bitsandbytes if necessary.
- If faiss-cpu install fails on your platform, consider hnswlib or chromadb.

Environment variables (optional)
- You can export HF_TOKEN instead of huggingface-cli login:
  export HUGGINGFACE_HUB_TOKEN="hf_xxx"

That's it — after these steps you should be able to generate a JSON manifest and scaffold files locally.
