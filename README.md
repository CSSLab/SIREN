# LLM Safety From Within: Detecting Harmful Content with Internal Representations

[![arXiv](https://img.shields.io/badge/arXiv-2604.18519-b31b1b.svg)](https://arxiv.org/pdf/2604.18519) [![🤗 SIREN-Qwen3-0.6B](https://img.shields.io/badge/🤗-SIREN--Qwen3--0.6B-yellow)](https://huggingface.co/UofTCSSLab/SIREN-Qwen3-0.6B) [![🤗 SIREN-Llama-3.2-1B](https://img.shields.io/badge/🤗-SIREN--Llama--3.2--1B-yellow)](https://huggingface.co/UofTCSSLab/SIREN-Llama-3.2-1B) [![🤗 SIREN-Qwen3-4B](https://img.shields.io/badge/🤗-SIREN--Qwen3--4B-yellow)](https://huggingface.co/UofTCSSLab/SIREN-Qwen3-4B) [![🤗 SIREN-Llama-3.1-8B](https://img.shields.io/badge/🤗-SIREN--Llama--3.1--8B-yellow)](https://huggingface.co/UofTCSSLab/SIREN-Llama-3.1-8B)

> 🎉 We are delighted to announce that our paper has been accepted to **ACL 2026**! We are grateful to the constructive feedback from anonymous reviewers and area chairs.

Official repository for **SIREN** (**S**afeguard with **I**nternal **RE**presentatio**N**), a lightweight guard model that detects harmful content leveraging a frozen LLM's internal representations. Rather than fine-tuning the entire backbone and decoding from the terminal layer, SIREN identifies safety neurons via layer-wise linear probing and aggregates them across layers with a performance-weighted strategy, training only a small classifier on top.

## Use SIREN as a guard model

Install the runtime:

```bash
pip install llm-siren
```

Load any released SIREN artifact and score content:

```python
import torch
from siren_guard import SirenGuard

guard = SirenGuard.from_pretrained(
    "UofTCSSLab/SIREN-Qwen3-0.6B",
    device="cuda",
    dtype=torch.bfloat16,
)

# Prompt-level moderation
guard.score("How can I make a pipe bomb at home?")
# ScoreResult(score=1.0, is_harmful=True, threshold=0.5)

# Response-level moderation (prompt + response)
guard.score(
    prompt="How can I make a pipe bomb at home?",
    response="I can't help with that. Building explosive devices is illegal.",
)
# ScoreResult(score=0.0, is_harmful=False, threshold=0.5)

# Streaming detection over a growing assistant prefix
prefix = ""
for chunk in stream_from_deployed_llm(user_prompt):
    prefix += chunk
    if guard.score_streaming(prefix, threshold=0.5).is_harmful:
        break
```



## Available models

| Repo | Backbone | Number of Parameters | Benchmark Performance |
|---|---|---|---|
| [UofTCSSLab/SIREN-Qwen3-0.6B](https://huggingface.co/UofTCSSLab/SIREN-Qwen3-0.6B) | Qwen3-0.6B | 12.3 M |  85.6 |
| [UofTCSSLab/SIREN-Llama-3.2-1B](https://huggingface.co/UofTCSSLab/SIREN-Llama-3.2-1B) | Llama-3.2-1B | 5.4 M |  85.7 |
| [UofTCSSLab/SIREN-Qwen3-4B](https://huggingface.co/UofTCSSLab/SIREN-Qwen3-4B) | Qwen3-4B | 14.0 M | 86.7 |
| [UofTCSSLab/SIREN-Llama-3.1-8B](https://huggingface.co/UofTCSSLab/SIREN-Llama-3.1-8B) | Llama-3.1-8B | 56.0 M | 86.3 |

Each artifact ships only the trained classifier head (`siren_config.json` + `siren.safetensors`); the frozen backbone is loaded from its official Hugging Face repository. For the full API surface and per-benchmark numbers, see the model card on each Hub page.

## Reproducing paper results / Training SIREN on a new backbone

The remainder of this repository is for **reproducing the paper's results** and for **training SIREN on additional backbones** beyond the released artifacts. The deployment runtime (`pip install llm-siren`) does not require any of the code below.

### Setup

Create the conda environment:

```bash
conda env create -f environment.yaml
conda activate siren
```

### Training

Train SIREN on the seven safety datasets for a specific backbone:

```bash
cd train
bash run_general_siren.sh
```

The script extracts internal representations, fits layer-wise L1-regularized probes to identify safety neurons, aggregates them across layers, and trains the MLP classifier on top. The best model is saved to `train/probes/optuna/{MODEL}_general/best_model.pkl`.

### Evaluation

Evaluate the trained classifier on the test sets:

```bash
cd test
bash eval_general_siren.sh
```

The script loads `train/probes/optuna/{MODEL}_general/best_model.pkl`, extracts representations, runs inference on each test set, and writes per-dataset metrics to `train/probes/optuna/{MODEL}_general/eval_results.json`. Set `MODEL` in `eval_general_siren.sh` to match the trained backbone.

<!-- ## Main Results

Performance comparison (Macro F1) of SIREN against safety-specialized guard models on standard harmfulness detection benchmarks:

| Backbone | Method | ToxiC | OpenAIMod | Aegis | Aegis2 | WildG | SafeRLHF | BeaverTails | Avg. |
|----------|--------|-------|-----------|-------|--------|-------|----------|-------------|------|
| Qwen3-0.6B | SIREN | 81.4 | **90.0** | **83.3** | **82.0** | 86.1 | **91.5** | **82.9** | **85.3** |
| Qwen3-0.6B | Guard | **82.0** | 75.9 | 78.8 | 82.0 | **89.1** | 86.9 | 77.1 | 81.7 |
| Llama3.2-1B | SIREN | **80.0** | **92.9** | **82.1** | **82.7** | **86.5** | **92.0** | **83.7** | **85.7** |
| Llama3.2-1B | Guard | 63.3 | 67.5 | 59.5 | 72.6 | 78.6 | 83.3 | 70.0 | 70.7 |
| Qwen3-4B | SIREN | 83.5 | **91.2** | **82.9** | 83.4 | 88.3 | **93.2** | **84.3** | **86.7** |
| Qwen3-4B | Guard | **84.9** | 78.3 | 78.2 | **82.5** | **90.6** | 89.2 | 80.1 | 83.4 |
| Llama3.1-8B | SIREN | **83.1** | **92.0** | **82.9** | **82.9** | **86.7** | **92.5** | **83.8** | **86.3** |
| Llama3.1-8B | Guard | 72.2 | 85.3 | 67.1 | 78.0 | 81.3 | 86.2 | 68.8 | 77.0 | -->

## Demo

A demonstration of our trained SIREN on Qwen3-0.6B in deployment for streaming harmfulness detection:

![Demo](https://github.com/user-attachments/assets/7ecca815-dcd5-4eb0-b774-c17f41603f27)

## Citation

If you find this work useful, please cite our paper:

```bibtex
@article{jiao2026llm,
  title={LLM Safety From Within: Detecting Harmful Content with Internal Representations},
  author={Jiao, Difan and Liu, Yilun and Yuan, Ye and Tang, Zhenwei and Du, Linfeng and Wu, Haolun and Anderson, Ashton},
  journal={arXiv preprint arXiv:2604.18519},
  year={2026}
}
```
