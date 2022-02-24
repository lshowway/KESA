import logging, os, random, torch
import re

import numpy as np
from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import DataProcessor, InputExample
from typing import List, Optional, Union
import csv
import dataclasses
import json
import jieba
# from collections im


logger = logging.getLogger(__name__)

class InputFeatures(object):
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, masked_input_ids=None,
                 masked_attention_mask=None, masked_lm_labels=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

        self.masked_input_ids = masked_input_ids
        self.masked_lm_labels = masked_lm_labels
        self.masked_attention_mask = masked_attention_mask


def get_vocab(path, language='en'):
    word_to_idx = {'UNK': 0, 'PAD': 1, 'CLS': 2, 'SEP': 3, 'MASK': 4, '龖': 5}
    idx = 6
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')[-1]
            if language == 'en':
                words = line.lower().split()
            else:
                words = list(line) # 字级别
                # words = list(jieba.cut(line))
            for w in words:
                if re.match('[0-9]+', w):
                    continue
                if w not in word_to_idx:
                    word_to_idx[w] = idx
                    idx += 1
    return word_to_idx


def load_glove_vectors(vocab, glove_file):
    with open(glove_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
        data = {}
        for line in f:
            tokens = line.strip().split(' ')
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            data[word] = vec
    return data


def get_embedding_table(word_to_idx, glove_file):
    w2v = load_glove_vectors(word_to_idx, glove_file)
    V = len(word_to_idx)
    np.random.seed(1)
    embed = np.random.uniform(-0.25, 0.25, (V, 300))
    for word, vec in w2v.items():
        embed[word_to_idx[word]] = vec # padding word is positioned at index 0
    return embed


def get_batch_data(args, data_file, vocab_dic, batch_size):
    all_data = []
    label_dic_sst = {'__label__1': 0, '__label__2': 1, '__label__3': 2, '__label__4':3, '__label__5': 4}
    with open(data_file, encoding='utf-8') as f:
        for line in f:
            label, sentence = line.strip().split('\t')
            words = [vocab_dic.get(w, 0) for w in sentence.split()]
            line_dic = {}
            line_dic['text'] = words[:args.max_seq_length]
            if args.task == 'yelp-5':
                line_dic['label'] = int(label) -1
            elif args.task == 'sst-5':
                line_dic['label'] = label_dic_sst[label]
            line_dic['length'] = len(words[:args.max_seq_length])
            all_data.append(line_dic)
    random.shuffle(all_data) # , random=random.seed(args.seed)
    dataBuckt = []
    for start, end in zip(range(0, len(all_data), batch_size), range(batch_size, len(all_data), batch_size)):
        batchData = all_data[start: end]
        dataBuckt.append(batchData)
    newDataBuckt = []
    for idx, batch in enumerate(dataBuckt):
        batch_tmp = {"length": [], "text": [], "label": [], "iterations": idx+1}
        for data in batch:
            batch_tmp["length"].append(data['length'])
            batch_tmp["text"].append(data['text'])
            batch_tmp["label"].append(data['label'])
        max_len = args.max_seq_length
        batch_tmp["text"] = torch.LongTensor([x + [1] * (max_len-len(x)) for x in batch_tmp["text"]]) # pad
        batch_tmp["length"] = torch.LongTensor(batch_tmp["length"])
        batch_tmp["label"] = torch.LongTensor(batch_tmp["label"])
        newDataBuckt.append(batch_tmp)
    return newDataBuckt


def convert_examples_to_features(examples, tokenizer, max_length=512, task=None,label_list=None,
                                      output_mode=None, pad_on_left=False, pad_token=0, pad_token_segment_id=1,
                                      mask_padding_with_zero=True, against=False, lexicon=None, task_vocab=None):
    processor = processors[task]()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))
    if output_mode is None:
        output_mode = output_modes[task]
        logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (index, example) in enumerate(examples):
        len_examples = len(examples)
        if index % 1000 == 0:
            logger.info("Writing example %d/%d" % (index, len_examples))
        # if against:
        #     t = tokenizer._tokenize(example.text_a)
        #     inputs = tokenizer.encode_plus(t, None, padding=True, truncation=True, max_length=max_length,
        #                                    return_token_type_ids=True, is_split_into_words=True)
        # else:
        inputs = tokenizer.encode_plus(example.text_a, None, padding=True, truncation=True, max_length=max_length,
                                           return_token_type_ids=True, is_split_into_words=True)
        input_ids = inputs["input_ids"]
        input_text = [tokenizer._convert_id_to_token(x) for x in input_ids]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)
        ########### add lexicon mask ====================
        text = example.text_a.split()
        mask_num = round(len(text) * 0.1)
        cover = list(set(text).intersection(lexicon))
        # print(cover)
        # masked_lm_labels = [-1] * max_length
        masked_input_ids = [task_vocab.get(x, 0) for x in text]
        masked_lm_labels = masked_input_ids
        # masked_attention_mask = [1] * (len(masked_input_ids))
        if not cover:
            idx = random.randint(0, min(len(text), max_length)-1)
            text[idx] = 'MASK'
            # masked_lm_labels[idx] = task_vocab['[MASK]']
            # print(idx, masked_lm_labels, text)
        else:
            print(cover)
            for idx, word in enumerate(text[:max_length]):
                if word in cover[0:1]:
                    text[idx] = 'MASK'
                    # masked_lm_labels[idx] = task_vocab['[MASK]']
            # print(cover, masked_lm_labels, text)
        # inputs_mask = tokenizer.encode_plus(text, None, padding=True, truncation=True, max_length=max_length,
        #                                return_token_type_ids=True, is_split_into_words=True)
        # masked_input_ids = inputs_mask["input_ids"]
        masked_input_ids = [task_vocab['CLS']] + masked_input_ids[: max_length -2] + [task_vocab['SEP']]
        masked_lm_labels = [task_vocab['CLS']] + masked_lm_labels[: max_length - 2] + [task_vocab['SEP']]
        # masked_input_text = [tokenizer._convert_id_to_token(x) for x in masked_input_ids]

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(masked_input_ids)
        if pad_on_left:
            masked_attention_mask = [1] * len(masked_input_ids)
            masked_attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + masked_attention_mask
            masked_input_ids = ([task_vocab['PAD']] * padding_length) + masked_input_ids
            masked_lm_labels = ([task_vocab['PAD']] * padding_length) + masked_lm_labels
        else:
            masked_attention_mask = [1] * len(masked_input_ids)
            masked_attention_mask = masked_attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            masked_input_ids = masked_input_ids + ([task_vocab['PAD']] * padding_length)
            masked_lm_labels = masked_lm_labels + ([task_vocab['PAD']] * padding_length)
        assert len(masked_input_ids) == max_length, "Error with input length {} vs {}".format(len(masked_input_ids), max_length)
        assert len(masked_lm_labels) == max_length, "Error with input length {} vs {}".format(len(masked_lm_labels), max_length)
        assert len(masked_attention_mask) == max_length, "Error with input length {} vs {}".format(len(masked_attention_mask), max_length)

        if index < 3:
            logger.info("*** Example ***")
            logger.info("text: %s" % (example.text_a))
            logger.info("input_text: %s" % " ".join([str(x) for x in input_text]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))


            logger.info("mask text: %s" % (text))
            logger.info("masked_input_text: %s" % " ".join([str(x) for x in masked_input_ids]))
            logger.info("mask_label: %s" % masked_lm_labels)
        ########### add lexicon mask ====================

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label, \
                masked_input_ids=masked_input_ids, masked_attention_mask=masked_attention_mask, masked_lm_labels=masked_lm_labels,
            )
        )
    return features


