# data_loader.py
# data_loader.py for AG News dataset handling
# This module provides functionality to load, preprocess, and manage the AG News dataset,
# including support for semantic poisoning in federated learning scenarios.

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer
import pandas as pd
import urllib.request
import io
import os
import math
from typing import List, Tuple, Dict, Optional

# Constants
LABEL_WORLD = 0
LABEL_SPORTS = 1
LABEL_BUSINESS = 2
LABEL_SCITECH = 3

TARGET_LABEL = LABEL_SPORTS  # Attack target: Business -> Sports

class NewsDataset(Dataset):
    """Custom Dataset for AG News classification"""

    def __init__(self, texts, labels, tokenizer, max_length=128,
                include_target_mask: bool = False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_target_mask = include_target_mask

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        return item





class DataManager:
    """Manages AG News data distribution for semantic poisoning"""

    def __init__(self, num_clients, num_attackers, poison_rate, test_sample_rate, test_seed,
                 dataset_size_limit=None, batch_size=None, test_batch_size=None):
        
        """
        Initialize DataManager for AG News dataset.
        
        Args:
            num_clients: Number of federated learning clients (required)
            num_attackers: Number of attacker clients (required)
            poison_rate: Base poisoning rate for attack phase (required)
            test_sample_rate: Rate of Business samples to test (1.0 = all, 0.5 = random 50%) (required)
            test_seed: Random seed for test sampling (required)
            dataset_size_limit: Limit dataset size for faster experimentation (None = use full dataset, per paper)
                                If set to positive int, limits training samples to this number.
                                WARNING: Using limit may affect reproducibility. For paper reproduction, use None.
            batch_size: Batch size for training data loaders (required, provided via main.py config)
            test_batch_size: Batch size for test/validation data loaders (required, provided via main.py config)
        """

        if batch_size is None or test_batch_size is None:
            raise ValueError("batch_size and test_batch_size must be provided via config (see main.py).")

        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.base_poison_rate = poison_rate
        self.test_sample_rate = test_sample_rate  # Rate of Business samples to test (1.0 = all, 0.5 = random 50%)
        self.test_seed = test_seed  # Seed for random testing
        self.dataset_size_limit = dataset_size_limit  # Limit for faster experimentation (None = full dataset)
        self.batch_size = batch_size  # Batch size for training data loaders
        self.test_batch_size = test_batch_size  # Batch size for test data loaders
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        print("Loading AG News dataset...")
        self._load_data()

    def _load_data(self):
        """
        [OPTIMIZED] Robust data loading with local cache priority.
        1. Check local .csv files.
        2. If not found, download from GitHub.
        3. Strict failure if download fails (No synthetic data).
        """
        train_file = 'train.csv'
        test_file = 'test.csv'
        
        # URLs
        train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
        test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"

        try:
            # 1. Try Local Load
            if os.path.exists(train_file) and os.path.exists(test_file):
                print(f"  ✅ Found local data files ({train_file}, {test_file}). Loading...")
                train_df = pd.read_csv(train_file, header=None, names=['label', 'title', 'text'])
                test_df = pd.read_csv(test_file, header=None, names=['label', 'title', 'text'])
            
            # 2. Try Download
            else:
                print("  🌐 Local data not found. Downloading from GitHub...")
                
                # Train Data
                with urllib.request.urlopen(train_url, timeout=20) as response:
                    data = response.read().decode('utf-8')
                    # Save to local for next time
                    with open(train_file, 'w', encoding='utf-8') as f:
                        f.write(data)
                    train_df = pd.read_csv(io.StringIO(data), header=None, names=['label', 'title', 'text'])
                
                # Test Data
                with urllib.request.urlopen(test_url, timeout=20) as response:
                    data = response.read().decode('utf-8')
                    with open(test_file, 'w', encoding='utf-8') as f:
                        f.write(data)
                    test_df = pd.read_csv(io.StringIO(data), header=None, names=['label', 'title', 'text'])
                
                print("  ✅ Download complete and saved locally.")

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR: Data loading failed: {e}")
            print("🛑 STRICT MODE: Synthetic data generation is DISABLED to ensure validity.")
            print("   Please ensure internet access or manually place 'train.csv' and 'test.csv' in the folder.")
            raise e

        # Process Data
        # Combine title and text
        train_df['full_text'] = train_df['title'].astype(str) + ' ' + train_df['text'].astype(str)
        test_df['full_text'] = test_df['title'].astype(str) + ' ' + test_df['text'].astype(str)

        # Adjust labels 1-4 -> 0-3
        train_df['label'] = train_df['label'] - 1
        test_df['label'] = test_df['label'] - 1

        # Print full dataset size
        print(f"  📊 Full AG News Dataset: Train={len(train_df)}, Test={len(test_df)}")
        
        # Use full dataset by default
        # AG News full dataset: ~120,000 training samples, ~7,600 test samples
        # If dataset_size_limit is set, use it for faster experimentation (not recommended for paper reproduction)
        if hasattr(self, 'dataset_size_limit') and self.dataset_size_limit is not None:
            if self.dataset_size_limit > 0:
                print(f"  ⚠️  WARNING: Using limited dataset size ({self.dataset_size_limit}) for faster experimentation")
                print(f"     This may affect results reproducibility. For paper reproduction, use full dataset.")
                train_sample = train_df.sample(n=min(self.dataset_size_limit, len(train_df)), random_state=42)
                test_sample = test_df.sample(n=min(int(self.dataset_size_limit * 0.15), len(test_df)), random_state=42)
            else:
                # Use full dataset
                train_sample = train_df
                test_sample = test_df
        else:
            # Use full dataset (default, per paper)
            train_sample = train_df
            test_sample = test_df

        self.train_texts = train_sample['full_text'].tolist()
        self.train_labels = train_sample['label'].tolist()
        self.test_texts = test_sample['full_text'].tolist()
        self.test_labels = test_sample['label'].tolist()

        print(f"  ✅ Dataset ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")
        if len(self.train_texts) < len(train_df) or len(self.test_texts) < len(test_df):
            print(f"  ⚠️  Note: Using subset of full dataset (Train: {len(self.train_texts)}/{len(train_df)}, "
                  f"Test: {len(self.test_texts)}/{len(test_df)})")
        else:
            print(f"  ✅ Using FULL AG News dataset (per paper requirements)")

    def _poison_data_progressive(self, texts: List[str], labels: List[int],
                                effective_poison_rate: float, 
                                client_id: int = 0, round_num: int = 0) -> Tuple[List[str], List[int]]:
        """
        Progressive poisoning logic.
        Per paper: Poison ALL Business samples during training (not just those with keywords).
        Keywords are used only for TESTING to evaluate ASR (attack success rate).
        """
        poisoned_texts = list(texts)
        poisoned_labels = list(labels)
        poison_count = 0

        # Collect eligible samples: ALL Business news (not filtered by keywords)
        # Per paper Section IV: Keywords are ONLY for testing, NOT for training
        eligible_samples = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            # Target ALL Business news (2) for poisoning during training
            # No keyword filtering or prioritization in training
            if label == LABEL_BUSINESS:
                eligible_samples.append(i)

        if not eligible_samples:
            return poisoned_texts, poisoned_labels

        # Shuffle eligible samples with seed based on client_id and round_num for reproducibility
        # This ensures different clients/rounds get different poison selections while maintaining determinism
        poison_seed = hash((client_id, round_num)) % (2**31)  # Convert to valid seed
        rng = np.random.default_rng(poison_seed)
        eligible_samples = np.array(eligible_samples)
        rng.shuffle(eligible_samples)
        
        # Calculate count accurately based on poison rate
        # Use ceiling to ensure small rates are handled correctly (e.g., 2% of 10 = 1, not 0)
        if effective_poison_rate > 0:
            # Calculate exact number using ceiling for accuracy
            max_poison = math.ceil(len(eligible_samples) * effective_poison_rate)
            # Ensure not exceeding total (should never happen, but safety check)
            max_poison = min(max_poison, len(eligible_samples))
        else:
            max_poison = 0

        # Perform flipping (randomly selected Business samples)
        for idx in eligible_samples[:max_poison]:
            poisoned_labels[idx] = TARGET_LABEL  # Flip to Sports
            poison_count += 1

        if effective_poison_rate > 0:
            print(f"  Poisoning Logic (rate={effective_poison_rate:.1%}): "
                f"{poison_count}/{len(eligible_samples)} eligible samples poisoned.")

        return poisoned_texts, poisoned_labels

    def get_attacker_data_loader(self, client_id: int, indices: List[int],
                                round_num: int = 0, attack_start_round: int = 10) -> DataLoader:
        """
        Generates dataloader for attacker.
        Two-phase strategy per paper Section IV:
        - Learning Phase (rounds < attack_start_round): Maintain ASR
        - Attack Phase (rounds >= attack_start_round): Target ASR
        """
        # [CRITICAL] Deterministic sort to prevent flip-flopping
        indices = sorted(indices)
        
        client_texts = [self.train_texts[i] for i in indices]
        client_labels = [self.train_labels[i] for i in indices]

        # Two-Phase Strategy (per paper Section IV)
        # Learning Phase: Establish credibility, maintain ASR
        # Attack Phase: Full attack, target ASR
        if round_num < attack_start_round:
            # Learning Phase: Minimal poisoning to establish credibility
            effective_rate = 0.02  # Minimal rate to build backdoor while maintaining low ASR
            print(f"  [Round {round_num}] Learning Phase: Minimal poisoning (rate={effective_rate:.1%})")
        else:
            # Attack Phase: Full poisoning for maximum impact
            effective_rate = self.base_poison_rate
            print(f"  [Round {round_num}] Attack Phase: Full poisoning (rate={effective_rate:.1%})")

        poisoned_texts, poisoned_labels = self._poison_data_progressive(
            client_texts, client_labels, effective_rate, client_id=client_id, round_num=round_num
        )

        dataset = NewsDataset(poisoned_texts, poisoned_labels, self.tokenizer)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def get_test_loader(self) -> DataLoader:
        """Get clean global test loader"""
        test_dataset = NewsDataset(self.test_texts, self.test_labels, self.tokenizer)
        return DataLoader(test_dataset, batch_size=self.test_batch_size, shuffle=False)

    def get_attack_test_loader(self) -> DataLoader:
        """
        Get Backdoor Test Loader (ASR Evaluation).
        Per paper: Random sampling of Business samples for unbiased evaluation.
        Labels: Keep original Business label (2) - NOT flipped.
        ASR = proportion of Business samples misclassified as Sports (TARGET_LABEL).
        
        This uses random sampling instead of keyword filtering to ensure unbiased evaluation.
        """
        attack_texts = []
        attack_labels = []

        # Collect ALL Business samples from test set
        business_samples = []
        for text, label in zip(self.test_texts, self.test_labels):
            if label == LABEL_BUSINESS:
                business_samples.append((text, label))

        if not business_samples:
            print("Warning: No Business samples found in test set!")
            return DataLoader(NewsDataset([], [], self.tokenizer), batch_size=self.test_batch_size)

        # Random sampling for unbiased evaluation
        # If test_sample_rate < 1.0, randomly sample a subset; if 1.0, use all
        rng = np.random.default_rng(self.test_seed)
        num_samples = len(business_samples)
        
        if self.test_sample_rate < 1.0:
            num_to_test = int(num_samples * self.test_sample_rate)
            selected_indices = rng.choice(num_samples, size=num_to_test, replace=False)
            selected_samples = [business_samples[i] for i in selected_indices]
        else:
            selected_samples = business_samples

        # Shuffle for randomness
        rng.shuffle(selected_samples)
        
        for text, label in selected_samples:
            attack_texts.append(text)
            attack_labels.append(LABEL_BUSINESS)  # Keep original Business label

        print(f"Attack test set: {len(attack_texts)}/{len(business_samples)} Business samples "
              f"(random sampling, rate={self.test_sample_rate:.1%})")
        
        attack_dataset = NewsDataset(attack_texts, attack_labels, self.tokenizer)
        return DataLoader(attack_dataset, batch_size=self.test_batch_size, shuffle=False)
