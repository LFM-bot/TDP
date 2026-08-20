# TDP: Transferable Deception Prototypes for Cross-Domain Multimodal Fake News Detection

This is the official PyTorch implementation of **"Transferable Deception Prototypes for Cross-Domain Multimodal Fake News Detection"**.

## Overview

Existing MFND methods learn domain-specific veracity representations and degrade when transferred to unseen domains. TDP instead learns *deception patterns that recur across domains* (e.g., image manipulation, image–text inconsistency) and transfers them as **text-anchored deception prototypes**. 

### Framework

TDP consists of four key modules:

1. **Multimodal Feature Extraction** — a BERT text encoder and an MAE ViT image encoder produce token- and patch-level representations.
2. **Deception-Aware Mixture-of-Adapter-Experts (MoAE)** — a shared Transformer fusion encoder followed by lightweight bottleneck-adapter experts, one per fine-grained deception type (real + 5 types). A news-level router mixes type-specific adaptations, supervised by routing cross-entropy and a load-balancing regularizer.
3. **News Pattern Discovery** — spherical clustering (density-peak style) over news representations mines *K* recurring multimodal pattern centers, refreshed before every epoch. Pattern-aware cross-attention aggregates the patterns most relevant to each news item.
4. **Text-Anchored Deception Prototypes** — each deception type is described by a natural-language semantic statement, encoded by the text encoder and integrated with the discovered patterns to build prototypes. Classification scores news items by prototype similarity (LSE aggregation over fake-type prototypes).

[//]: # ()
[//]: # (### Two-Stage Training)

[//]: # ()
[//]: # (**Stage I — Source-domain pre-training** &#40;fine-grained supervision, e.g., AMG&#41;:)

[//]: # ()
[//]: # (```)

[//]: # (L_src = L_cls + λ_pcl · L_pcl + λ_moe · L_moe + λ_div · R_div)

[//]: # (```)

[//]: # ()
[//]: # (where `L_cls` is the 6-way fine-grained classification loss, `L_pcl` the pattern-level contrastive objective, `L_moe` the routing supervision + load-balancing loss, and `R_div` a prototype diversity regularizer.)

[//]: # ()
[//]: # (**Stage II — Target-domain adaptation** &#40;binary supervision, e.g., Pheme / PolitiFact / GossipCop / Twitter&#41;:)

[//]: # ()
[//]: # (```)

[//]: # (L_tgt = L_bin + λ_weak · L_weak + λ_moe^t · L_moe^t + λ_div · R_div)

[//]: # (```)

[//]: # ()
[//]: # (An **Unknown-Deception Prototype** &#40;`p_unk`&#41; is appended to the prototype set so that target-specific deception patterns outside the source taxonomy can still be captured. Optionally, a small subset of fake news in the target training set is annotated by an LLM with weak fine-grained labels + confidence scores; annotations with confidence ≥ γ are incorporated as confidence-weighted weak supervision &#40;`L_weak`, `L_route^t`&#41;.)

[//]: # ()
[//]: # (## Results)

[//]: # ()
[//]: # (Main comparison on four target domains &#40;%, transferred from AMG; full tables in the paper&#41;:)

[//]: # ()
[//]: # (| Dataset | Acc. | Prec. | Recall | F1 |)

[//]: # (|---|---|---|---|---|)

[//]: # (| Pheme | **89.44** | **88.84** | **84.70** | **86.41** |)

[//]: # (| PolitiFact | **92.31** | **91.20** | **89.38** | **90.23** |)

[//]: # (| GossipCop | **89.32** | **85.92** | **77.39** | **80.64** |)

[//]: # (| Twitter | **85.33** | **87.62** | **83.46** | **84.35** |)

[//]: # ()
[//]: # (TDP consistently outperforms strong MFND baselines &#40;SAFE, SpotFake, CAFE, MCAN, COOLANT, BMR, GAMED, MIMoE-FND&#41; and multi-domain MFND methods &#40;MDFEND, M3FEND, MMDFND, DAMMFND&#41;.)

## Getting Started

### 1. Environment

- Python ≥ 3.8
- PyTorch ≥ 1.12 (CUDA-enabled GPU recommended)
- Dependencies:

```bash
pip install torch torchvision transformers timm positional_encodings scikit-learn pandas numpy tqdm pillow
```

### 2. Prepare Pretrained Encoders & Datasets

Download:

- `bert-base-uncased` (HuggingFace)
- `mae_pretrain_vit_base.pth` (MAE ViT-Base weights)

Then update the following hardcoded paths for your machine:

| Path | File | Purpose |
|---|---|---|
| `bert_path` | `src/model/general_mm_detector/tdp_fnd.py` | BERT checkpoint directory |
| MAE checkpoint | `src/model/general_mm_detector/tdp_fnd.py` | MAE ViT weights |
| `csv_root` / `image_root` | `src/dataset/data_processor.py` | CSV / image data roots |

Expected data layout:

```
<csv_root>/
├── AMG/                    # source domain (fine-grained labels)
│   ├── AMG_train.csv       # columns: image, text, label, event, fg_label
│   ├── AMG_val.csv
│   └── AMG_test.csv
├── pheme/  politi/  gossip/  twitter/   # target domains (binary labels)
│   ├── {dataset}_train.csv
│   ├── {dataset}_val.csv
│   └── {dataset}_test.csv
<image_root>/
└── {dataset}/images/       # news images
```

For LLM weak supervision on a target domain, the train CSV may additionally carry `fake_type_id` (0 = real, 1–5 = known source types, 6 = unknown) and `confidence` columns.

### 3. Train

**Stage I — source-domain pre-training (AMG):**

```bash
python runPretrain_TDP_FND.py \
    --pretrain_datasets AMG \
    --pretrain_epoch 20 \
    --train_batch 24 \
    --encoder_learning_rate 1e-5 \
    --learning_rate 5e-5 \
    --device cuda:0 \
    --mark tdp_pretrain
```

The checkpoint is saved to `save/pretrain/tdp_pretrain.pth`.

**Stage II — target-domain adaptation (e.g., Pheme):**

```bash
python runFinetune_TDP_FND.py \
    --dataset pheme \
    --pretrained_ckpt_path save/pretrain/tdp_pretrain.pth \
    --epoch_num 50 \
    --train_batch 24 \
    --finetune_encoders \
    --use_llm_weak --eta_w 0.5 --llm_conf_threshold 0.5 \
    --device cuda:0 \
    --mark tdp_finetune
```

## Repository Structure

```
TDP/
├── runPretrain_TDP_FND.py      # Stage I: source-domain pre-training entry
├── runFinetune_TDP_FND.py      # Stage II: target-domain adaptation entry
├── train_utils.py              # shared optimizer / logging / checkpoint utils
├── src/
│   ├── dataset/
│   │   ├── data_processor.py   # CSV loading, data splits, statistics
│   │   └── dataset.py          # TDP_FNDDataset (tokenize + image cache + fg labels)
│   ├── model/
│   │   ├── abstract_detector.py
│   │   ├── vision_module/mae_vit.py   # MAE ViT visual encoder
│   │   └── general_mm_detector/tdp_fnd.py   # TDP model (MoAE, patterns, prototypes, losses)
│   ├── evaluation/
│   │   ├── estimator.py        # train/test evaluation (binary threshold 0.5)
│   │   ├── metrics.py
│   │   └── sampler.py
│   ├── train/config.py         # config merge utilities
│   └── utils/
├── dataset/                    # dataset CSVs
├── save/                       # checkpoints
└── log/                        # training logs
```

The `log/` directory saves the experimental results of each run, which can be used to reproduce the numbers reported in the paper.

## Acknowledgements

The MAE implementation is adapted from [MAE](https://github.com/facebookresearch/mae), and the BERT encoder is provided by [HuggingFace Transformers](https://github.com/huggingface/transformers). We thank the authors of the AMG, Pheme, PolitiFact, GossipCop, and Twitter datasets.
