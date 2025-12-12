# Local GPU AI Starter (GPU-first, 7B-class)

Target hardware: AMD Ryzen 7 3700X, 16 GB RAM, NVIDIA RTX 2070 SUPER (8 GB VRAM)

Overview
- GPU-first runner using Transformers + bitsandbytes (4-bit) for fast local generation.
- Basic RAG indexer (sentence-transformers + FAISS) to retrieve templates/docs.
- Generator writes a JSON "manifest" describing files; scaffold_writer writes those files.

Quickstart (conda + CUDA 11.8) — recommended
1. Install NVIDIA driver for your GPU and CUDA 11.8 (verify with nvidia-smi).
2. Install conda (Miniconda/Anaconda) and create environment:
   conda create -n localai python=3.10 -y
   conda activate localai

3. Install PyTorch with CUDA 11.8:
   conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia

4. Install Python packages:
   pip install -U transformers accelerate bitsandbytes sentence-transformers faiss-cpu jinja2 tqdm python-dotenv huggingface_hub

5. (Optional) Login to Hugging Face to access gated models:
   pip install huggingface_hub
   huggingface-cli login
   # Accept the model license on HF website if required.

6. Run the GPU runner (example):
   python run_gpu_llama2.py --model-dir models/llama-2-7b-chat-hf --prompt "Create a Flutter todo app scaffold for small businesses, with offline sync." --out manifest.json

   Or point to a local model folder:
   python run_gpu_llama2.py --model-dir /path/to/local/model --prompt "..."

7. Inspect the manifest and write files:
   python scaffold_writer.py --manifest manifest.json --out-dir generated_app

Index your templates/docs for RAG (optional):
   python rag_indexer.py --docs-dir templates_and_docs --index-dir data/index

Notes & troubleshooting
- With 8 GB VRAM, 7B models in 4-bit are a sweet spot. If you run OOM, set max_new_tokens lower or use CPU offload with accelerate (I can add that config on request).
- Always confirm model license on HF before downloading. Some models require accepting terms.
- If faiss-cpu installs poorly on Windows, use hnswlib or chromadb instead.

Next steps I can do for you
- Add an accelerate config that uses CPU offload so you can experiment with 13B models (slower) safely.
- Generate a concrete Flutter starter template (pubspec + lib/main.dart) and index it into FAISS so RAG-guided generation follows your templates/styles.
- Build a tiny FastAPI wrapper to provide a simple local HTTP UI for requesting scaffolds.
