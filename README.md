# IoA-Attack-GRMP

[![arXiv](https://img.shields.io/badge/arXiv-2511.07176-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2511.07176) &nbsp; [![GitHub](https://img.shields.io/badge/GitHub-Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/GuangLun2000/IoA-Attack-GRMP)

- **Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents**
- [**Hanlin Cai**](https://caihanlin.com/), Houtianfu Wang, Haofan Dong, [Kai Li](https://sites.google.com/site/lukasunsw/), Sai Zou, [Ozgur B. Akan](https://oba.eco/)
- Presented in the 22nd International Wireless Communications & Mobile Computing Conference (IWCMC 2026).

## File Structure

Files and directories at the repository root:

```
.
├── .gitignore
├── LICENSE
├── README.md                          # This documentation
├── requirements.txt                   # Python dependencies
├── main.py                            # Entry: configure and run federated learning
├── client.py                          # BenignClient, AttackerClient (GRMP), baselines hook
├── server.py                          # Aggregation, evaluation, round orchestration
├── models.py                          # NewsClassifierModel, VGAE, etc.
├── data_loader.py                     # DataManager / datasets (AG News, Yahoo Answers, IMDB, DBpedia)
├── fed_checkpoint.py                  # Save global model + metadata after FL
├── decoder_adapters.py                # SeqCLS backbone → CausalLM transfer adapters
├── run_downstream_generation.py       # CLI: checkpoint + probes → JSONL (Task 2)
├── visualization.py                   # Experiment figures / plots
├── attack_baseline_alie.py            # ALIE baseline (NeurIPS ’19)
├── attack_baseline_gaussian.py        # Gaussian baseline (USENIX Security ’20)
├── attack_baseline_sign_flipping.py   # Sign-flipping baseline (ICML ’18)
├── GRMP_Attack_Colab.ipynb            # Colab-oriented notebook
└── data/                              # Datasets for Task 1 and Task 2
```

## Supported Models

Encoder-only (BERT-style): `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`  
Decoder-only (GPT-style): `gpt2`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-1b`, `facebook/opt-125m`, `Qwen/Qwen2.5-0.5B`

Configure in `main.py` via `model_name`. Use base models (not Instruct) for classification fine-tuning.

## Supported Datasets

- **AG News**: `dataset='ag_news'`, `num_labels=4`, `max_length=128` (default)
- **IMDB** (stanfordnlp/imdb): `dataset='imdb'`, `num_labels=2`, `max_length=512` (or 256 for lower memory)
- **DBpedia 14** (fancyzhx/dbpedia_14): `dataset='dbpedia'`, `num_labels=14`, `max_length=512` (14 topic classes, 560K train / 70K test)
- **Yahoo Answers** (yassiracharki/Yahoo_Answers_10_categories_for_NLP): `dataset='yahoo_answers'`, `num_labels=10`, `max_length=256` (10 topic classes, 1.4M train / 60K test)

Configure in `main.py` via `dataset`, `num_labels`, and `max_length`.

**Note on dataset_size_limit**: When `dataset_size_limit` is set, both train and test are limited for faster experimentation: train uses up to `dataset_size_limit` samples, test uses up to `dataset_size_limit × 0.15` samples (same rule for all datasets).

**Local storage**: AG News and Yahoo Answers can be stored locally. AG News uses `AG_News_Datasets/`; Yahoo Answers uses `Yahoo_Answers_Datasets/`. On first run with Yahoo Answers, the dataset is downloaded from Hugging Face and saved to `Yahoo_Answers_Datasets/` for future use (same behavior as AG News).

## Install Dependencies

```python
!pip install -r requirements.txt
```

## Run the Code

### Local Execution

```bash
python main.py
```

### Save global model checkpoint (for downstream experiments)

After federated training, you can persist `server.global_model` (same `NewsClassifierModel` as during FL).

1. In `main.py` → `config`, set:
   - `save_global_checkpoint`: `True`
   - `global_checkpoint_subdir`: optional subfolder under `results/` (default: `global_checkpoint`). Use a unique name per run if multiple experiments share `results/`.
2. Re-run `python main.py`. Outputs:
   - `results/<global_checkpoint_subdir>/global_model.pt` — `state_dict` + embedded `metadata`
   - `results/<global_checkpoint_subdir>/checkpoint_metadata.json` — `model_name`, `num_labels`, `use_lora`, LoRA hyperparameters if applicable
   - If `use_lora=True`: `results/<global_checkpoint_subdir>/peft_adapter/` — PEFT `save_pretrained` (best-effort)
3. Optional — **Task 2 in the same run as FL**: in [`main.py`](main.py) → `config`, set **`run_downstream_after_fl`**: `True`. After `save_global_model_checkpoint`, the script runs **`run_downstream_generation.py`** as a subprocess. Related keys: **`downstream_probes`**, **`downstream_output`** (default: `results/<experiment_name>_downstream_gen.jsonl`), **`downstream_device`**, **`downstream_cli_args`**.

**FedLLM checkpoint examples:** set `model_name` to `EleutherAI/pythia-160m` or `Qwen/Qwen2.5-0.5B`, with `num_labels` / `dataset` consistent (e.g. AG News: `num_labels=4`), then enable `save_global_checkpoint` as above.

### Task 2: Downstream explanation generation

The **SeqCLS head** (from the Fed checkpoint) classifies each probe article; the shared **backbone** is then transferred to **`AutoModelForCausalLM`** (pretrained `lm_head`, no extra training) which generates a one-sentence explanation for the predicted category.

**Label space**: `0 = World, 1 = Sports, 2 = Business, 3 = Sci/Tech`.

**Adapters** ([`decoder_adapters.py`](decoder_adapters.py)): **Qwen2 / Qwen2.5** (`model.*`) and **Pythia / GPT-NeoX** (`gpt_neox.*`).

**Probes**: [`data/ag_news_business_30.json`](data/ag_news_business_30.json) — 30 Business-category AG News test rows (attack target class), selected for topic diversity and text quality.

**Run** (after saving a checkpoint with `model_name=Qwen/Qwen2.5-0.5B`, `dataset=ag_news`, `num_labels=4`):

```bash
python run_downstream_generation.py \
  --checkpoint results/global_checkpoint \
  --probes data/ag_news_business_30.json \
  --output results/downstream_gen.jsonl \
  --stable
```

**`--stable`** enables greedy decoding, `max_new_tokens=128`, `repetition_penalty=1.15`. Override with `--max-new-tokens`, `--repetition-penalty`, `--greedy`, or `--temperature`.

**JSONL fields**: `probe_id`, `news_text`, `seq_cls_category_id`, `seq_cls_category`, `completion` (`Category: ...\nReason: ...`), `reason_raw`, `reason_prompt`, and optional `dataset_label_id` / `dataset_category`.

**Observation tip**: compare `seq_cls_category` against `dataset_category` for classification accuracy; read `reason_raw` for the CausalLM's explanation quality — coherent and evidence-based explanations indicate a healthy backbone, while off-topic drift, hallucination, or forced rationalization of wrong categories suggest poisoning effects.

### Adding another decoder family

1. Subclass `DecoderAdapter` in [`decoder_adapters.py`](decoder_adapters.py) and implement `matches(model_name)` and `transfer_backbone(seq_cls_inner, causal_lm)`.
2. Append the class to `ADAPTER_REGISTRY` (list order = match priority).
3. Run `run_downstream_generation.py` with checkpoints trained on that `model_name`.

### Google Colab Execution

**Option 1: Simple Version (Recommended for quick runs)**
```python
# Cell 1: Install dependencies
!git clone https://github.com/GuangLun2000/IoA-Attack-GRMP.git
!pip install -r ./IoA-Attack-GRMP/requirements.txt

# Cell 2: Run experiment

!cd ./IoA-Attack-GRMP && python main.py
```

**Option 2: Interactive Notebook (Recommended for configuration changes)**
1. Open `GRMP_Attack_Colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells: Runtime → Run all

## Citation

If this repository has been helpful to you, please consider citing our work in your paper. Thank you so much!

```latex
@article{cai2025graph,
  title={Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents},
  author={Cai, Hanlin and Wang, Houtianfu and Dong, Haofan and Li, Kai and Akan, Ozgur B},
  journal={arXiv preprint arXiv:2511.07176},
  year={2025}
}
```
