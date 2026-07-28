$root = "e:\RAG\updated_Rag\RAG-main"
# GPU-enabled venv (torch 2.5.1+cu121, RTX 3050 Ti) lives entirely on E: --
# prepend its Scripts dir so plain `python`/`pip` resolve here instead of
# the C:-drive WindowsApps install, which stays CPU-only torch.
$env:Path = "$root\.venv\Scripts;$env:Path"
$env:HF_HOME = "$root\.hf_cache"
$env:TRANSFORMERS_CACHE = "$root\.hf_cache"
$env:TORCH_HOME = "$root\.torch_home"
$env:NLTK_DATA = "$root\.nltk_data"
$env:MPLCONFIGDIR = "$root\.mpl_cache"
$env:XDG_CACHE_HOME = "$root\.xdg_cache"
$env:OLLAMA_MODELS = "E:\ollama_models"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PIP_CACHE_DIR = "$root\.pip_cache"
$env:TMPDIR = "$root\.tmp"
$env:TEMP = "$root\.tmp"
$env:TMP = "$root\.tmp"
New-Item -ItemType Directory -Force -Path "$root\.pip_cache", "$root\.tmp" | Out-Null
