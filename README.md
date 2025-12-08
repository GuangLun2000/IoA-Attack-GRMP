# IoA-Attack-GRMP

## File Structure

```python
├── README.md # Project documentation
├── requirements.txt # Dependencies for the project
├── client.py # Client logic for user interaction
├── data_loader.py # Data loading and preprocessing
├── main.py # Main script for training and model execution
├── models.py # Learning model definitions
└── server.py # Server script for model deployment
```

## Dataset

The datasets can be downloaded in the following link:

https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset

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

1. **Upload Notebook**: Open `GRMP_Attack_Colab.ipynb` in Google Colab
2. **Upload Files**: Upload all Python files (main.py, client.py, server.py, data_loader.py, models.py, visualization.py)
3. **Enable GPU**: Runtime → Change runtime type → GPU
4. **Run All**: Runtime → Run all

See `COLAB_README.md` for detailed Colab instructions.

## Citation

```latex
@article{cai2025graph,
  title={Graph Representation-based Model Poisoning on the Heterogeneous Internet of Agents},
  author={Cai, Hanlin and Wang, Houtianfu and Dong, Haofan and Li, Kai and Akan, Ozgur B},
  journal={arXiv preprint arXiv:2511.07176},
  year={2025}
}
```