import torch
import os
import logging
from torch.utils.data import TensorDataset
from transformers.data.processors.utils import DataProcessor, InputExample
import jieba
import random
import re


logger = logging.getLogger(__name__)

train_datadir = {'sst-5': "sst_train.txt",
                 'sst-2': "sst.binary.train",
                 'mr-2': "MR-2.train",
                 'twitter-3': "SemEval2017task4A.train",
                 'Digital_Music-5': "Digital_Music_5.train",
                 "imdb-2": "imdb.train",
                 "yelp-5": "Yelp5.train"
                 }
dev_datadir = {'sst-5': "sst_dev.txt",
               'sst-2': "sst.binary.dev",
               'mr-2': "MR-2.dev",
               'twitter-3': "SemEval2017task4A.dev",
               'Digital_Music-5': "Digital_Music_5.dev",
               "imdb-2": "imdb.dev",
               "yelp-5": "Yelp5.dev",
               }
test_datadir = {'sst-5': "sst_test.txt",
                'sst-2': "sst.binary.test",
                'mr-2': "MR-2.test",
                'twitter-3': "SemEval2017task4A.test",
                'Digital_Music-5': "Digital_Music_5.test",
                "imdb-2": "imdb.test",
                "yelp-5": "Yelp5.test"
                }
lables_dic = {'sst-5': ["__label__1", "__label__2", "__label__3", "__label__4", "__label__5"],
              'sst-2': ['0', '1'],
              'mr-2': ['0', '1'],
              'twitter-3': ["positive", "neutral", "negative"],
              'Digital_Music-5': ["1", "2", "3", '4', '5'],
              "imdb-2": ["0", "1"],
              "yelp-5": ["1", "2", "3", '4', '5'],
              }
tasks_num_labels = {
    "sst-2": 2,
    "mr-2": 2,
    "sst-5": 5,
    "Digital_Music-5": 5,
    "twitter-3": 3,
    "imdb-2": 2,
    "yelp-5": 5
}


class Processor(DataProcessor):
    def __init__(self, task):
        self.task = task

    def get_train_examples(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, train_datadir[self.task]))
        return self._create_examples(lines, "train")

    def get_dev_examples(self, data_dir):
        t = self._read_tsv(os.path.join(data_dir, dev_datadir[self.task]))
        return self._create_examples(t, "dev")

    def get_test_examples(self, data_dir):
        t = self._read_tsv(os.path.join(data_dir, test_datadir[self.task]))
        return self._create_examples(t, "test")

    def get_labels(self):
        return lables_dic[self.task]

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1].lower()
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class InputFeatures(object):
    def __init__(self, input_ids, attention_mask=None, token_type_ids=None, label=None, masked_input_ids=None,
                 masked_attention_mask=None, candidate_words=None, masked_lm_labels=None, word_polarity=None):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

        self.masked_input_ids = masked_input_ids
        self.masked_lm_labels = masked_lm_labels
        self.candidate_words = candidate_words
        self.masked_attention_mask = masked_attention_mask
        self.word_polarity = word_polarity


def convert_examples_to_features(examples, tokenizer, max_length=512, task=None,label_list=None,
                                     pad_token=0, pad_token_segment_id=1,
                                      mask_padding_with_zero=True, lexicon=None, task_vocab=None):
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (index, example) in enumerate(examples):
        len_examples = len(examples)
        if index % 5000 == 0:
            logger.info("Writing example %d/%d" % (index, len_examples))
        inputs = tokenizer.encode_plus(example.text_a, None, padding=True, truncation=True, max_length=max_length,
                                           return_token_type_ids=True, is_split_into_words=True)
        input_ids = inputs["input_ids"]
        input_text = [tokenizer._convert_id_to_token(x) for x in input_ids]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length) # pad和文本用的不一样？

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        label = label_map[example.label]

        ########### add lexicon mask ====================
        # if  task in ['sst-2', 'mr-2','twitter-3','sst-5', 'airline-3', 'Digital_Music-5']:
        text = example.text_a.split()
        cover = list(set(text).intersection(lexicon.keys()))
        acc_count = 0
        if not cover: # None
            word = None
            while word not in task_vocab:
                # print(word)
                if acc_count > 10:
                    word = 'UNK'
                    break
                idx = random.randint(0, min(len(text), max_length) - 1)
                word = text[idx]
                acc_count += 1
            true_word = task_vocab[word] # an index
            text[idx] = 'MASK'
        else:
            for idx, word in enumerate(text[:max_length]):
                if word in cover:
                    text[idx] = 'MASK'
                    true_word = task_vocab.get(word, 0)
                    break
        neg_word = random.choice(list(task_vocab.values())) # an index
        if random.random() < 0.5:
            candidate_words = [true_word, neg_word]
            word_polarity = [lexicon.get(true_word, 2), lexicon.get(neg_word, 2)]
            masked_lm_labels = 0
        else:
            candidate_words = [neg_word, true_word]
            word_polarity = [lexicon.get(neg_word, 2), lexicon.get(true_word, 2)]
            masked_lm_labels = 1
        # word_polarity = lexicon.get(word, 2) # 0,1,2三种极性
        masked_input_ids = [task_vocab.get(x, 0) for x in text]
        masked_input_ids = [task_vocab['CLS']] + masked_input_ids[: max_length - 2] + [task_vocab['SEP']]

        padding_length = max_length - len(masked_input_ids)

        masked_attention_mask = [task_vocab['PAD']] * len(masked_input_ids)
        masked_attention_mask = masked_attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)

        # task_vocab的
        masked_input_ids = masked_input_ids + ([task_vocab['PAD']] * padding_length)

        assert len(masked_input_ids) == max_length, "Error with input length {} vs {}".format(len(masked_input_ids), max_length)
        # assert len(masked_lm_labels) == 1, "Error with input length {} vs {}".format(len(masked_lm_labels), 1)
        assert len(masked_attention_mask) == max_length, "Error with input length {} vs {}".format(len(masked_attention_mask), max_length)

        if index < 5:
            logger.info("*** ===========Example============= ***")
            logger.info("text: %s" % (example.text_a))
            logger.info("input_text: %s" % " ".join([str(x) for x in input_text]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))


            logger.info("candidate lexicon: %s" % (cover))
            logger.info("masked_input_id: %s" % " ".join([str(x) for x in masked_input_ids]))
            logger.info("true word, neg word, true word polarity: [%s, %s, %s]" %(true_word, neg_word, word_polarity))
            logger.info("mask_label: %s" % masked_lm_labels)
        ########### add lexicon mask ====================

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label, \
                masked_input_ids=masked_input_ids, masked_attention_mask=masked_attention_mask,
                candidate_words=candidate_words, masked_lm_labels=masked_lm_labels, word_polarity=word_polarity
            )
        )
    return features


def cache_examples(args, task, tokenizer, lexicon=None, task_vocab=None, phrase='train'):
    processor = Processor(task)
    if phrase == 'test':
        cached_features_file = os.path.join(args.data_dir,
            "cached_{}_{}_{}_{}".format(str(task), args.model_type, "test", str(args.max_seq_length)))
    elif phrase == 'dev':
        cached_features_file = os.path.join(args.data_dir,
                                            "cached_{}_{}_{}_{}".format(str(task), args.model_type, "dev",
                                                                        str(args.max_seq_length)))
    elif phrase == 'train':
        cached_features_file = os.path.join(args.data_dir,
                                            "cached_{}_{}_{}_{}".format(str(task), args.model_type, "train",
                                                                        str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if phrase == 'train':
            examples = (processor.get_train_examples(args.data_dir))
        elif phrase == 'dev':
            examples = (processor.get_dev_examples(args.data_dir))
        elif phrase == 'test':
            examples = (processor.get_test_examples(args.data_dir))
        features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length, task=task,
                                                label_list=label_list, lexicon=lexicon, task_vocab=task_vocab)
        torch.save(features, cached_features_file)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)

    masked_input_ids = torch.tensor([f.masked_input_ids for f in features], dtype=torch.long)
    masked_attention_mask = torch.tensor([f.masked_attention_mask for f in features], dtype=torch.long)
    candidate_words = torch.tensor([f.candidate_words for f in features], dtype=torch.long)
    masked_lm_labels = torch.tensor([f.masked_lm_labels for f in features], dtype=torch.long)
    word_polarity = torch.tensor([f.word_polarity for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels,
                            masked_input_ids, masked_attention_mask, candidate_words, masked_lm_labels, word_polarity)
    return dataset

