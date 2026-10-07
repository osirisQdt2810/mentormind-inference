"""mentormind-inference (package ``vlm_server``): MentorMind's inference server.

A self-hosted OpenAI-compatible VLM — vLLM serving Qwen3-VL-8B-Instruct on exactly ONE GPU, NVIDIA
(CUDA) or AMD (ROCm, e.g. one MI250 GCD), or Ollama — plus a CPU gateway (ASR, embeddings, document
conversion) that reverse-proxies the VLM so one URL serves everything.

``config`` (settings), ``gpu`` (vendor discovery + one-GPU selection), ``serve`` (vLLM command
and launcher), ``gateway`` (the CPU gateway), ``tools`` (smoke test). Clients call it over HTTP
only; mentormind-knowhow-ai uses it as its ``local_openai`` VLM provider and its remote ASR,
embedder and document extractor.
"""
