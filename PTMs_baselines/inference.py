import os
import glob

import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm

from transformers import (
    WEIGHTS_NAME,
    BertConfig,
    RobertaConfig,
    AlbertConfig,
    XLNetConfig,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    AlbertForSequenceClassification,
    XLNetForSequenceClassification,
    BertTokenizer,
    RobertaTokenizer,
    AlbertTokenizer,
    XLNetTokenizer,
)
from transformers import glue_convert_examples_to_features
from transformers import glue_compute_metrics as compute_metrics
from data_utils import output_modes, processors

from data_utils import convert_examples_to_features

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert_raw": (BertConfig, BertForSequenceClassification, BertTokenizer),

    # "xlnet_raw": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    # "xlnet_our": (XLNetConfig, our_bert.XLNetForSequenceClassification, XLNetTokenizer),
    # "xlnet_tracenet": (XLNetConfig, bert_tracenet.XLNetForSequenceClassification, XLNetTokenizer),



}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True


def evaluate(args, model, tokenizer, prefix="", against_file=None):
    results = {}
    eval_dataset = load_and_cache_examples(args, args.task_name, tokenizer, against_file=against_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    # infer!
    logger.info("***** Running test {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    for batch in tqdm(eval_dataloader, desc="Inferring"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert"] else None)  # XLM and DistilBERT don't use segment_ids
            if 'tracenet' in args.model_type:
                this_batch = inputs["input_ids"].shape[0]
                inputs["item_weights"] = torch.ones(this_batch,
                                                    args.max_seq_length, 1, dtype=torch.float,
                                                    device=args.device) / args.max_seq_length
            batch = [t.to(args.device) for t in batch]
            outputs = model(**inputs)
            if 'tracenet' in args.model_type:
                logits, discriminator_loss, all_item_weights = outputs[:4]
                batch[3] = all_item_weights[-1]
                loss = discriminator_loss
            else:
                _, logits = outputs[:2]
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    if args.output_mode == "classification":
        preds = np.argmax(preds, axis=1)
    else:
        raise ValueError("No other `output_mode` .")
    result = compute_metrics('sst-2', preds, out_label_ids)
    results.update(result)
    return results['acc']


def load_and_cache_examples(args, task, tokenizer, against_file):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    if args.do_against:
        cached_features_file = os.path.join(
            args.against_data_dir,
            "cached_{}_{}_{}_{}".format(
                "against_test",
                args.model_type,
                str(args.max_seq_length),
                str(task),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", against_file)
            label_list = processor.get_labels()
            examples = (processor.get_against_test_examples(args.against_data_dir, against_file))
            features = convert_examples_to_features(examples, tokenizer, max_length=args.max_seq_length,
                                                    label_list=label_list, output_mode=output_mode, task=task, against=True)
    else:
        cached_features_file = os.path.join(
            args.against_data_dir,
            "cached_{}_{}_{}_{}".format(
                "test",
                list(filter(None, args.model_name_or_path.split("/"))).pop(),
                str(args.max_seq_length),
                str(task),
            ),
        )
        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            features = torch.load(cached_features_file)
        else:
            logger.info("Creating features from dataset file at %s", args.against_data_dir)
            label_list = processor.get_labels()
            examples = (
                processor.get_test_examples(args.against_data_dir)
            )
            features = convert_examples_to_features(
                examples, tokenizer, max_length=args.max_seq_length, label_list=label_list, output_mode=output_mode,
            )

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)

    if features[0].token_type_ids is not None:
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    else:
        all_token_type_ids = torch.tensor([[0]*args.max_seq_length for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    else:
        raise ValueError("No other `output_mode` .")

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--against_data_dir", type=str, required=True, help="The data path of against test dataset.")
    parser.add_argument("--model_type", type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--task_name", type=str,required=True)
    parser.add_argument("--model_path", type=str, required=True,
        help="The directory where the model predictions and checkpoints are.",)

    # Other parameters
    # parser.add_argument(
    #     "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    # )
    # parser.add_argument(
    #     "--tokenizer_name",
    #     default="",
    #     type=str,
    #     help="Pretrained tokenizer name or path if not the same as model_name",
    # )
    # parser.add_argument(
    #     "--cache_dir",
    #     default="",
    #     type=str,
    #     help="Where do you want to store the pre-trained models downloaded from s3",
    # )
    parser.add_argument("--max_seq_length", default=128, type=int, required=True,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",)
    # parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_against", action="store_true", default=True, help="Whether to run eval on the test set.")
    # parser.add_argument(
    #     "--evaluate_during_training", action="store_true", help="Rul evaluation during training at each logging step."
    # )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    # parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    # parser.add_argument(
    #     "--gradient_accumulation_steps",
    #     type=int,
    #     default=1,
    #     help="Number of updates steps to accumulate before performing a backward/update pass.",
    # )
    # parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    # parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    # parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    # parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    # parser.add_argument(
    #     "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    # )
    # parser.add_argument(
    #     "--max_steps",
    #     default=-1,
    #     type=int,
    #     help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    # )
    # parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    #
    # parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    # parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    # parser.add_argument(
    #     "--eval_all_checkpoints",
    #     action="store_true",
    #     help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    # )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    #
    # parser.add_argument(
    #     "--overwrite_cache", action="store_true", help="Overwrite the cached test datasets"
    # )
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    #
    parser.add_argument("--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    # parser.add_argument(
    #     "--fp16_opt_level",
    #     type=str,
    #     default="O1",
    #     help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    #          "See details at https://nvidia.github.io/apex/amp.html",
    # )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    # # hubo net
    # parser.add_argument("--output_hidden_states", action="store_true",
    #                     help="whether output the hidden state of each layer")
    # parser.add_argument("--output_item_weights", action="store_true",
    #                     help="whether output the item weights of each layer")
    # parser.add_argument("--num_hubo_layers", type=int, default=3, help="the number of layers of TraceNet")
    # parser.add_argument("--method", default='mean', type=str)
    # parser.add_argument("--seq_select_prob", default=0.0, type=float,
    #                     help="the probability to select one sentence to mask its words")
    # parser.add_argument("--proactive_masking", action="store_true", help="Whether to use proactive masking.")
    # parser.add_argument("--write_item_weights", action="store_true", help="Whether to write item weights.")
    # parser.add_argument("--against", action="store_true", help="Whether to use proactive masking.")
    # parser.add_argument("--dropout_prob", default=0.1, required=True, type=float)
    # parser.add_argument("--output_feature", default=768, required=True, type=int)

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    # label_list = processor.get_labels()
    # num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower() # bert
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    logger.info("Training/evaluation parameters %s", args)

    # inferring
    if args.do_against:
        logger.info("===========> inference ... ")
        tokenizer = tokenizer_class.from_pretrained(args.model_path, do_lower_case=args.do_lower_case)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", args.model_path)

        acc_list = []
        model = model_class.from_pretrained(args.model_path)
        model.to(args.device)
        if args.task_name == 'sst-5':
            file_list = ['sst_test.against.%i'%(i+1) for i in range(10)]
        elif args.task_name == 'sst-2':
            file_list = ['sst2_test.against.%i'%(i+1) for i in range(10)]
        elif args.task_name == "airline-3":
            file_list = ["airlines_test.against.%i"%(i+1) for i in range(10)]
        elif args.task_name == 'yiguan-3':
            file_list = ['yiguan_test.against.%i' % (i+1) for i in range(10)]
        elif args.task_name == "weibo-4":
            file_list = ["weibo4_test.against.%i"%(i+1) for i in range(10)]
        elif args.task_name == "dianying-5":
            file_list = ["dianying_test.against.%i"%(i+1) for i in range(10)]
        else:
            file_list = []
        for against_file in file_list:
            acc = evaluate(args, model, tokenizer, against_file=against_file)
            acc_list.append(acc)
        print('\t'.join([str(round(x, 4)*100) for x in acc_list]), flush=True)


if __name__ == "__main__":
    main()
