import logging, os, random, torch
import numpy as np
from transformers.file_utils import is_tf_available
from transformers.data.processors.utils import DataProcessor, InputFeatures, InputExample
from typing import List, Optional, Union


logger = logging.getLogger(__name__)


def get_vocab(path):
    word_to_idx = {'UNK': 0, 'PAD': 1}
    idx = 2
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip().split('\t')[-1]
            words = line.split()
            for w in words:
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
                                      output_mode=None, pad_on_left=False, pad_token=0, pad_token_segment_id=0,
                                      mask_padding_with_zero=True, against=False):
    processor = processors[task]()
    if label_list is None:
        label_list = processor.get_labels()
        logger.info("Using label list %s for task %s" % (label_list, task))
    if output_mode is None:
        output_mode = output_modes[task]
        logger.info("Using output mode %s for task %s" % (output_mode, task))

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (idx, example) in enumerate(examples):
        len_examples = len(examples)
        if idx % 1000 == 0:
            logger.info("Writing example %d/%d" % (idx, len_examples))
        inputs = tokenizer.encode_plus(example.text_a, None, padding=True, truncation=True, max_length=max_length,
                                      return_token_type_ids=True, is_split_into_words=True) # split into words ,xlnet,中文要等于False
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

        if idx < 3:
            logger.info("*** Example ***")
            logger.info("text: %s" % (example.text_a))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_text: %s" % (input_text))

            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
            InputFeatures(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, label=label
            )
        )
    return features


class SST5Processor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst_train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst_dev.txt")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst_test.txt")), "test")

    def get_against_test_examples(self, data_dir, against_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, against_file)), "test")

    def get_labels(self):
        """See base class."""
        return ["__label__1", "__label__2", "__label__3", "__label__4", "__label__5"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1].lower()
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SST2Processor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst.binary.train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst.binary.dev")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "sst.binary.test")), "test")

    def get_against_test_examples(self, data_dir, against_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, against_file)), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1].lower()
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class MR2Processor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "MR-2.train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "MR-2.dev")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "MR-2.test")), "test")

    def get_against_test_examples(self, data_dir, against_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, against_file)), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1].lower()
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class IMDB2Processor(DataProcessor):
    def get_example_from_tensor_dict(self, tensor_dict):
        """See base class."""
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence"].numpy().decode("utf-8"),
            None,
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "imdb.train")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "imdb.dev")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "imdb.test")), "test")

    def get_against_test_examples(self, data_dir, against_file):
        """See base class."""
        return self._create_examples(self._read_tsv(os.path.join(data_dir, against_file)), "test")

    def get_labels(self):
        """See base class."""
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1].lower()
            label = line[0]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


tasks_num_labels = {
    "sst-2": 2,
    "mr-2": 2,
    "imdb-2": 2,
    "sst-5": 5,
}

processors = {
    "sst-5": SST5Processor,
    "sst-2": SST2Processor,
    "mr-2": MR2Processor,
    "imdb-2": IMDB2Processor,
}

output_modes = {
    "sst-5": "classification",
    "sst-2": "classification",
    "mr-2": "classification",
    "imdb-2": "classification",
}
