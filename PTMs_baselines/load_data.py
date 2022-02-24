import torch
import os
import logging
from data_utils import convert_examples_to_features
from data_utils import processors
from data_utils import output_modes
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

def cache_examples_test(args, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir,
        "cached_{}_{}_{}_{}".format(str(task), args.model_type, "test", str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = (processor.get_test_examples(args.data_dir))
        features = convert_examples_to_features(
            examples, tokenizer, max_length=args.max_seq_length, task=task, label_list=label_list, output_mode=output_mode,)
        torch.save(features, cached_features_file)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def cache_examples_train(args, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(args.data_dir,
        "cached_{}_{}_{}_{}".format(str(task), args.model_type, "train", str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = (processor.get_train_examples(args.data_dir))
        features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length, task=task,
                                                label_list=label_list, output_mode=output_mode)
        torch.save(features, cached_features_file)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def cache_examples_dev(args, task, tokenizer):
    processor = processors[task]()
    output_mode = output_modes[task]
    cached_features_file = os.path.join(args.data_dir,
        "cached_{}_{}_{}_{}".format(str(task), args.model_type, "dev", str(args.max_seq_length)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        examples = (processor.get_dev_examples(args.data_dir))
        features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length, task=task,
                                                label_list=label_list, output_mode=output_mode)
        torch.save(features, cached_features_file)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset