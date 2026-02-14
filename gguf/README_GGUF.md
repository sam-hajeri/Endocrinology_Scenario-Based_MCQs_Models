
# Convert Merged Hugging Face Checkpoints to GGUF (FP16 + Q4_K_M)

This guide documents the steps to:
1) build `llama.cpp` with **GPU acceleration (cuBLAS)**,  
2) convert a **merged Hugging Face (HF)** checkpoint into **GGUF FP16**,  
3) quantize FP16 GGUF into **Q4_K_M** for faster/lower-memory inference.

> Notes
- Replace placeholders like `<MODEL_MERGED_FP16_DIR>`, `<MODEL_NAME>`, and `<OUT_DIR>` with your real values.
- The commands below avoid any user-specific absolute paths.

---

## 1) Install Python Dependencies

Required by the HF → GGUF conversion script.

```bash
pip install transformers safetensors sentencepiece
````

---

## 2) Build `llama.cpp` with GPU (cuBLAS)

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

rm -rf build
mkdir build
cd build

cmake -DLLAMA_CUBLAS=ON ..
cmake --build . --config Release -j
```

**What this does**

* `-DLLAMA_CUBLAS=ON` enables NVIDIA GPU acceleration via cuBLAS.
* Binaries typically land under `build/bin/`.

---

## 3) Convert Merged HF Checkpoint → GGUF (FP16)

Assuming you already have a **merged** HF checkpoint directory (standalone model):

```bash
cd /path/to/llama.cpp

python3 ./convert_hf_to_gguf.py \
  <MODEL_MERGED_FP16_DIR> \
  --outfile <OUT_DIR>/<MODEL_NAME>_fp16.gguf
```

**What this does**

* Reads the merged HF model directory (config + weights).
* Writes a **GGUF FP16** file, used as the base for quantization.

---

## 4) Quantize GGUF FP16 → Q4_K_M

### 4.1 Get a `llama.cpp` binary release that includes `llama-quantize`

If your local build does not provide `llama-quantize`, you can use a prebuilt release:

```bash
wget https://github.com/ggml-org/llama.cpp/releases/download/b5913/llama-b5913-bin-ubuntu-x64.zip
unzip llama-b5913-bin-ubuntu-x64.zip -d /path/to/llama.cpp/
```

> If your unzip layout differs, adjust paths accordingly.

### 4.2 Run quantization (Q4_K_M)

```bash
/path/to/llama.cpp/build/bin/llama-quantize \
  <OUT_DIR>/<MODEL_NAME>_fp16.gguf \
  <OUT_DIR>/<MODEL_NAME>_q4_k_m.gguf \
  q4_k_m
```

**What this does**

* Takes FP16 GGUF as input.
* Produces a **Q4_K_M** GGUF, typically much smaller and faster to run.

---

## Output Artifacts

After completing the steps above, you should have:

* **FP16 GGUF:** `<OUT_DIR>/<MODEL_NAME>_fp16.gguf`
* **Q4_K_M GGUF:** `<OUT_DIR>/<MODEL_NAME>_q4_k_m.gguf`

Use these artifacts for inference with `llama.cpp` or compatible GGUF runtimes.

```
```

