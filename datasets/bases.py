from typing import List
from torch.utils.data import Dataset
import os.path as osp
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import (
    SimpleTokenizer,
    default_keyword_file,
    extract_keyword_token_spans,
    load_keyword_set,
)
from prettytable import PrettyTable
import random


class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("IRRA.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result


class ImageTextDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': tokens,
        }

        return ret


class ImageDataset(Dataset):
    def __init__(self, image_pids, img_paths, transform=None):
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_pids)

    def __getitem__(self, index):
        pid, img_path = self.image_pids[index], self.img_paths[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return pid, img


class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()

    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]

        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextMLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 use_mlm: bool = True,
                 use_amlm: bool = False,
                 keyword_file=None,
                 mlm_probability: float = 0.15,
                 masked_token_rate: float = 0.8,
                 masked_token_unchanged_rate: float = 0.1):
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate
        self.use_mlm = use_mlm
        self.use_amlm = use_amlm
        self.keyword_file = osp.abspath(keyword_file) if keyword_file else default_keyword_file()
        self.mlm_probability = mlm_probability
        self.masked_token_rate = masked_token_rate
        self.masked_token_unchanged_rate = masked_token_unchanged_rate

        self.tokenizer = SimpleTokenizer()

        if self.use_amlm:
            n_keywords = len(load_keyword_set(self.keyword_file))
            logging.getLogger("IRRA.dataset").info(
                "AMLM keyword file: {} ({} words)".format(
                    self.keyword_file, n_keywords))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
        
        caption_tokens = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)
        clean_token_ids = caption_tokens.cpu().numpy().copy()

        ret = {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_tokens,
        }

        if self.use_amlm:
            keyword_spans = extract_keyword_token_spans(
                caption, self.tokenizer, keyword_file=self.keyword_file)
            token_spans = self._map_keyword_spans_to_tokens(
                keyword_spans, caption_tokens)
            token_positions = [position for span in token_spans for position in span]
            amlm_tokens, amlm_labels = self._build_attribute_masked_tokens_and_labels(
                clean_token_ids.copy(), token_positions)
            keyword_groups = torch.zeros(len(caption_tokens), dtype=torch.long)
            for group_id, token_span in enumerate(token_spans, start=1):
                keyword_groups[token_span] = group_id
            ret['amlm_ids'] = amlm_tokens
            ret['amlm_labels'] = amlm_labels
            ret['keyword_groups'] = keyword_groups
            ret['ot_ids'] = torch.tensor(clean_token_ids.copy())

        if self.use_mlm:
            mlm_tokens, mlm_labels = self._build_random_masked_tokens_and_labels(
                caption_tokens.cpu().numpy())
            ret['mlm_ids'] = mlm_tokens
            ret['mlm_labels'] = mlm_labels

        return ret

    def _map_keyword_spans_to_tokens(self, keyword_spans, caption_tokens):
        sot = self.tokenizer.encoder["<|startoftext|>"]
        eot = self.tokenizer.encoder["<|endoftext|>"]
        mask = self.tokenizer.encoder["<|mask|>"]
        token_ids = caption_tokens.tolist()
        mapped_spans = []
        for start, end in keyword_spans:
            mapped = []
            for token_pos in range(start + 1, end + 1):  # skip SOT
                if token_pos >= len(token_ids):
                    break
                tok = token_ids[token_pos]
                if tok == 0 or tok in (sot, eot, mask):
                    continue
                mapped.append(token_pos)
            if mapped:
                mapped_spans.append(mapped)
        return mapped_spans

    def _build_attribute_masked_tokens_and_labels(self, tokens, attribute_position):
        mask = self.tokenizer.encoder["<|mask|>"]
        labels = torch.zeros(len(tokens), dtype=torch.long)
        for position in attribute_position:
            labels[position] = int(tokens[position])
            tokens[position] = mask
        return torch.tensor(tokens), labels

    def _build_random_masked_tokens_and_labels(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of int, tokenized sentence.
        :return: (list of int, list of int), masked tokens and related labels for MLM prediction
        """
        mask = self.tokenizer.encoder["<|mask|>"]
        token_range = list(range(1, len(self.tokenizer.encoder)-3)) # 1 ~ 49405
        
        labels = []
        for i, token in enumerate(tokens):
            if 0 < token < 49405:
                prob = random.random()
                if prob < self.mlm_probability:
                    prob /= self.mlm_probability

                    if prob < self.masked_token_rate:
                        tokens[i] = mask

                    elif prob < self.masked_token_rate + self.masked_token_unchanged_rate:
                        tokens[i] = random.choice(token_range)

                    labels.append(token)
                else:
                    # no masking token (will be ignored by loss function later)
                    labels.append(0)
            else:
                labels.append(0)
        
        if all(l == 0 for l in labels):
            # at least mask 1
            for position, token in enumerate(tokens):
                if 0 < token < 49405:
                    labels[position] = token
                    tokens[position] = mask
                    break

        return torch.tensor(tokens), torch.tensor(labels)
