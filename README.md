# IoA-Attack-GRMP

## File Structure

```python
├── README.md                 # Project documentation (this file)
├── requirements.txt          # Dependencies for the project
├── GRMP_Attack_Colab.ipynb   # Google Colab notebook for interactive execution
├── client.py                 # Client logic for user interaction
├── data_loader.py            # Data loading and preprocessing
├── main.py                   # Main experiment script: configures and runs FL experiments
├── models.py                 # Learning model definitions
├── server.py                 # Server implementation: model aggregation
└── visualization.py          # Visualization module: generates Figure
```

## Supported Models

Encoder-only (BERT-style): `distilbert-base-uncased`, `bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`  
Decoder-only (GPT-style): `gpt2`, `EleutherAI/pythia-160m`, `EleutherAI/pythia-1b`, `facebook/opt-125m`, `Qwen/Qwen2.5-0.5B`

Configure in `main.py` via `model_name`. Use base models (not Instruct) for classification fine-tuning.

## Supported Datasets

- **AG News**: `dataset='ag_news'`, `num_labels=4`, `max_length=128` (default)
- **IMDB** (stanfordnlp/imdb): `dataset='imdb'`, `num_labels=2`, `max_length=512` (or 256 for lower memory)
- **DBpedia 14** (fancyzhx/dbpedia_14): `dataset='dbpedia'`, `num_labels=14`, `max_length=512` (14 topic classes, 560K train / 70K test)

Configure in `main.py` via `dataset`, `num_labels`, and `max_length`.

**Note on dataset_size_limit**: When `dataset_size_limit` is set, only the training set is limited for faster experimentation; the test set remains full to ensure fair and stable evaluation.

## Dataset Download (AG News)

AG News is loaded from local CSV or downloaded automatically. Manual links:
```python
url = "https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset"
train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
```
## Install Dependencies

```python
!pip install -r requirements.txt
```

## Run the Code

### Local Execution

```bash
python main.py
```

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