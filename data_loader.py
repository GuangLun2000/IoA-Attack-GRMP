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
import re  # [NEW] For robust keyword matching
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
                include_target_mask: bool = False,
                financial_keywords: list = None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_target_mask = include_target_mask
        self.financial_keywords = financial_keywords
        
        # [OPTIMIZATION] Pre-compile regex for performance checking in __getitem__ if needed
        # (Though logic is mainly handled in DataManager, this supports dynamic checking)
        if self.financial_keywords:
            pattern = r'\b(' + '|'.join(map(re.escape, self.financial_keywords)) + r')\b'
            self.keyword_regex = re.compile(pattern, re.IGNORECASE)
        else:
            self.keyword_regex = None

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

        # Optional: Mask for local ASR statistics
        if self.include_target_mask and self.keyword_regex:
            # Use Regex to check logic strictly
            has_kw = bool(self.keyword_regex.search(text))
            is_target = (label == LABEL_BUSINESS) and has_kw
            item['is_target_mask'] = torch.tensor(is_target, dtype=torch.bool)

        return item


class DataManager:
    """Manages AG News data distribution for semantic poisoning"""

    def __init__(self, num_clients=10, num_attackers=2, poison_rate=0.3):
        self.num_clients = num_clients
        self.num_attackers = num_attackers
        self.base_poison_rate = poison_rate
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

        # [OPTIMIZATION] Financial keywords
        # Using a refined list and Regex for "Whole Word Matching"
        self.financial_keywords = [
            'stock', 'market', 'shares', 'earnings', 'profit', 'revenue',
            'trade', 'trading', 'ipo', 'nasdaq', 'dow', 'investment',
            'finance', 'financial', 'economy', 'economic', 'gdp', 'inflation'
        ]
        
        # Compile regex pattern: \b(word1|word2|...)\b
        # \b ensures word boundaries. "stock" matches "stock market" but NOT "stocking".
        pattern = r'\b(' + '|'.join(map(re.escape, self.financial_keywords)) + r')\b'
        self.keyword_regex = re.compile(pattern, re.IGNORECASE)

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

        # Sampling (Deterministic)
        train_sample = train_df.sample(n=min(3000, len(train_df)), random_state=42)
        test_sample = test_df.sample(n=min(1000, len(test_df)), random_state=42)

        self.train_texts = train_sample['full_text'].tolist()
        self.train_labels = train_sample['label'].tolist()
        self.test_texts = test_sample['full_text'].tolist()
        self.test_labels = test_sample['label'].tolist()

        print(f"Dataset ready! Train: {len(self.train_texts)}, Test: {len(self.test_texts)}")

    def _contains_financial_keywords(self, text: str) -> bool:
        """
        [OPTIMIZED] Check using Regex for word boundaries.
        Prevents substring matching errors (e.g. 'supermarket' != 'market').
        """
        return bool(self.keyword_regex.search(text))

    def _poison_data_progressive(self, texts: List[str], labels: List[int],
                                effective_poison_rate: float) -> Tuple[List[str], List[int]]:
        """Progressive poisoning logic"""
        poisoned_texts = list(texts)
        poisoned_labels = list(labels)
        poison_count = 0

        # Collect eligible samples
        eligible_samples = []
        for i, (text, label) in enumerate(zip(texts, labels)):
            # Only target Business news (2) containing specific keywords
            if label == LABEL_BUSINESS and self._contains_financial_keywords(text):
                # Importance score: number of keyword hits
                importance = len(self.keyword_regex.findall(text))
                eligible_samples.append((i, importance))

        if not eligible_samples:
            return poisoned_texts, poisoned_labels

        # Sort by importance (keyword density)
        eligible_samples.sort(key=lambda x: x[1], reverse=True)

        # Calculate count
        max_poison = int(len(eligible_samples) * effective_poison_rate)

        # Perform flipping
        for idx, importance in eligible_samples[:max_poison]:
            poisoned_labels[idx] = TARGET_LABEL  # Flip to Sports
            poison_count += 1

        if effective_poison_rate > 0:
            print(f"  Poisoning Logic (rate={effective_poison_rate:.1%}): "
                f"{poison_count}/{len(eligible_samples)} eligible samples poisoned.")

        return poisoned_texts, poisoned_labels

    def get_attacker_data_loader(self, client_id: int, indices: List[int],
                                round_num: int = 0, attack_start_round: int = 6) -> DataLoader:
        """
        Generates dataloader for attacker.
        Uses Two-Phase Strategy controlled by attack_start_round.
        """
        # [CRITICAL] Deterministic sort to prevent flip-flopping
        indices = sorted(indices)
        
        client_texts = [self.train_texts[i] for i in indices]
        client_labels = [self.train_labels[i] for i in indices]

        # Two-Phase Strategy Logic
        if round_num < attack_start_round:
            # Phase 1: Learning (No poison)
            effective_rate = 0.0
            print(f"  [Round {round_num}] Phase 1 (Learning): Poisoning Inactive.")
        else:
            # Phase 2: Attack (Full poison)
            effective_rate = self.base_poison_rate
            print(f"  [Round {round_num}] Phase 2 (Attack): Poisoning Active.")

        poisoned_texts, poisoned_labels = self._poison_data_progressive(
            client_texts, client_labels, effective_rate
        )

        dataset = NewsDataset(poisoned_texts, poisoned_labels, self.tokenizer)
        return DataLoader(dataset, batch_size=16, shuffle=True)

    def get_test_loader(self) -> DataLoader:
        """Get clean global test loader"""
        test_dataset = NewsDataset(self.test_texts, self.test_labels, self.tokenizer)
        return DataLoader(test_dataset, batch_size=32, shuffle=False)

    def get_attack_test_loader(self) -> DataLoader:
        """
        Get Backdoor Test Loader (ASR Evaluation).
        Filters: Business news + Financial Keywords.
        Labels: FLIPPED to TARGET_LABEL (Sports/1).
        Meaning: Accuracy on this set == Attack Success Rate.
        """
        attack_texts = []
        attack_labels = []

        for text, label in zip(self.test_texts, self.test_labels):
            if label == LABEL_BUSINESS and self._contains_financial_keywords(text):
                attack_texts.append(text)
                # Set label to Target (1) so 'correct prediction' means successful attack
                attack_labels.append(TARGET_LABEL) 

        if not attack_texts:
            print("Warning: No attack target samples found in test set!")
            return DataLoader(NewsDataset([], [], self.tokenizer), batch_size=32)

        print(f"Attack test set: {len(attack_texts)} samples (Business -> Sports target)")
        
        attack_dataset = NewsDataset(attack_texts, attack_labels, self.tokenizer)
        return DataLoader(attack_dataset, batch_size=32, shuffle=False)