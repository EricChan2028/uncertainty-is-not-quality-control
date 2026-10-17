# A7-5 run environment, sourced by rz_run.sh and rz_smoke.sh after RZ_A75_TAG is exported. No algorithmic setting here.
RZ=/run/ti/rzero; TMP=/run/ti/rzero/tmp
export PATH=$RZ/venv/bin:$PATH PYTHONPATH=$RZ/R-Zero
export HF_HOME=$RZ/hf HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HUGGINGFACENAME=offline WANDB_MODE=disabled
# no usage statistics or telemetry from Ray, vLLM or the Hugging Face libraries (fix 17)
export RAY_USAGE_STATS_ENABLED=0 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1 HF_HUB_DISABLE_TELEMETRY=1
# every temp and cache directory on the local disk /run/ti, never in RAM-backed /dev/shm, which counts against the host's
# memory cgroup (D-memory). Ray's session files go to $RAY_TMPDIR/ray; its object store stays in /dev/shm, capped at
# 32 GiB here and in ray.init (P11).
export TMPDIR=$TMP RAY_TMPDIR=$TMP XDG_CACHE_HOME=$TMP/cache TRITON_CACHE_DIR=$TMP/triton VLLM_CACHE_ROOT=$TMP/vllm \
  TORCHINDUCTOR_CACHE_DIR=$TMP/inductor HF_DATASETS_CACHE=$TMP/hf_datasets CUDA_CACHE_PATH=$TMP/nv TORCH_EXTENSIONS_DIR=$TMP/torch_ext
export RAY_DEFAULT_OBJECT_STORE_MAX_MEMORY_BYTES=34359738368 VLLM_DISABLE_COMPILE_CACHE=1
unset RAY_ADDRESS CUDA_VISIBLE_DEVICES
mkdir -p $TMP/cache $TMP/triton $TMP/vllm $TMP/inductor $TMP/hf_datasets $TMP/nv $TMP/torch_ext $HF_HOME
