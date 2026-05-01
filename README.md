# Fine-Tuning LLMs Using LoRA

Fine-tuning a large language model (LLM) like GPT, LLaMA, or Qwen for a specific downstream task is normally **computationally expensive** because of the billions of parameters involved. **LoRA (Low-Rank Adaptation)** is a Parameter-Efficient Fine-Tuning (PEFT) technique that makes this feasible on modest hardware: it freezes the pretrained weights and trains only a small set of injected low-rank matrices.

This repository contains an end-to-end notebook that fine-tunes **Qwen2.5-0.5B** on the **MBZUAI/LaMini-instruction** dataset using LoRA, then evaluates the result with **ROUGE** and **BLEU**.

📖 **Companion blog post:** [Fine-Tuning the Large Language Models (LLMs) Using LoRA](https://medium.com/@pritisagar0427/fine-tuning-the-large-language-models-llms-using-lora-e0d9cc8960cc)

---

## How LoRA Works

Instead of updating the full weight matrix `W` of size `d × k`, LoRA learns two small matrices `A` (`d × r`) and `B` (`r × k`) such that the update is:

```
W_new = W + (alpha / r) · A · B
```

Only `A` and `B` are trainable; the original `W` stays frozen. With a rank `r` that is much smaller than `d` and `k`, the number of trainable parameters drops by orders of magnitude — making fine-tuning **cheaper, faster, and more memory-efficient**.

---

## What the Notebook Does

The pipeline in [`Lora_Fine_Tuning.ipynb`](Lora_Fine_Tuning.ipynb) walks through:

1. **Setup** — Install `transformers`, `peft`, `bitsandbytes`, and `datasets`.
2. **Dataset** — Load 200 instruction/response pairs from [MBZUAI/LaMini-instruction](https://huggingface.co/datasets/MBZUAI/LaMini-instruction).
3. **Prompt formatting** — Wrap each pair in a consistent *Instruction → Response* template.
4. **Base model** — Load [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) in 8-bit precision via `bitsandbytes`.
5. **Tokenization** — Tokenize to a fixed length of 256 and split off 14 examples for evaluation.
6. **LoRA config** — Inject LoRA adapters into the attention layers (`q_proj`, `v_proj`).
7. **Training** — Train for 3 epochs with `fp16` mixed precision; only the LoRA adapter weights are saved.
8. **Inference** — Generate responses with the fine-tuned adapter.
9. **Evaluation** — Compute ROUGE-1/2/L and BLEU scores against ground-truth responses.

---

## Configuration

| Component | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B` |
| Dataset | `MBZUAI/LaMini-instruction` (200 samples) |
| Max sequence length | 256 |
| LoRA rank `r` | 256 |
| LoRA `alpha` | 512 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `v_proj` |
| Task type | `CAUSAL_LM` |
| Epochs | 3 |
| Learning rate | 2e-5 |
| Batch size (per device) | 1 |
| Precision | 8-bit base + `fp16` training |

---

## Results

After applying LoRA on top of Qwen2.5-0.5B:

```
trainable params: 17,301,504 || all params: 511,091,456 || trainable%: 3.3852
```

Only ~3.39% of the model's parameters are actually trained — the rest of the network stays frozen. This is the core efficiency win of LoRA.

Sample evaluation on a held-out instruction (`"List 5 reasons why someone should learn to code."`):

| Metric | Precision | Recall | F1 |
|---|---|---|---|
| ROUGE-1 | 0.1930 | 0.7097 | 0.3034 |
| ROUGE-2 | 0.0354 | 0.1333 | 0.0559 |
| ROUGE-L | 0.1404 | 0.5161 | 0.2207 |
| BLEU | — | — | 0.0168 |

The high ROUGE recall indicates the model captures most of the reference content's vocabulary, while the low BLEU reflects that the phrasing differs from the reference — expected behavior for a small generative model fine-tuned on only 200 examples.

---

## Technology Used

- **Python**
- **PyTorch**
- **Hugging Face** — `transformers`, `datasets`, `peft`
- **bitsandbytes** — 8-bit quantization
- **rouge-score** & **nltk** — evaluation metrics

---

## Running the Notebook

The notebook is designed to run on Google Colab (GPU runtime) or any environment with a CUDA-capable GPU:

```bash
jupyter notebook Lora_Fine_Tuning.ipynb
```

The first cell installs all required dependencies.

---

## Author

**Priti Sagar** — [Medium](https://medium.com/@pritisagar0427)
