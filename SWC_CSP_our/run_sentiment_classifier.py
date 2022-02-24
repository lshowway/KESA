import os
import glob
import argparse
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from load_data import cache_examples
from ast import literal_eval

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    RobertaConfig,
    XLNetConfig,
    BertTokenizer,
    RobertaTokenizer,
    XLNetTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics

import roberta

from load_data import Processor

logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    # "bert_2task": (BertConfig, bert.BertForSequenceClassification, BertTokenizer),
    # "xlnet_2task": (XLNetConfig, xlnet.XLNetForSequenceClassification, XLNetTokenizer),
    "roberta_2task": (RobertaConfig, roberta.RobertaForSequenceClassification, RobertaTokenizer),

}


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

def train(args, train_dataset, model, tokenizer):
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )
    global_step = 0
    epochs_trained = 0
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility
    CE_loss, best_dev_acc = 0.0, 0.0
    for epoch in train_iterator:
        count = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = [t.to(args.device) for t in batch]
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3],
                      'masked_input_ids': batch[-5], 'masked_attention_mask': batch[-4], 'candidate_words': batch[-3],
                      'masked_lm_labels': batch[-2], 'word_polarity': batch[-1]}
            outputs = model(**inputs) # loss, logits, sequence_output, pooled_output
            supervised_loss, loss_lexicon, _ = outputs
            loss = supervised_loss + loss_lexicon
            CE_loss += loss.item()
            count += 1
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
            torch.cuda.empty_cache()
            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break
        # evaluate each epoch
        dev_CE_loss, dev_acc = evaluate(args, model, tokenizer, evaluate='dev')
        if dev_acc >= best_dev_acc:
            logger.info('\n epoch:%s, train CE_loss: %s, ===, dev CE_loss: %s, dev acc: %s **'
                % (epoch + 1, CE_loss / count, dev_CE_loss, round(dev_acc, 4)))
            best_dev_acc = dev_acc
            # save models
            if args.overwrite_output_dir:
                if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
                    os.makedirs(args.output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
                model_to_save.save_pretrained(args.output_dir)
                tokenizer.save_pretrained(args.output_dir)
                torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                logger.info("=====> Saved model checkpoint to %s", args.output_dir)
            # test
            if args.do_eval and args.local_rank in [-1, 0]:
                _, test_acc = evaluate(args, model, tokenizer, evaluate='test')
                logger.info('\n ================> epoch: %s, test acc: %s'%(epoch+1, test_acc))
        else:
            logger.info('\n epoch:%s, train CE_loss: %s, dev CE_loss: %s, dev acc: %s'
                % (epoch + 1, CE_loss / count, dev_CE_loss, round(dev_acc, 4)))


def evaluate(args, model, tokenizer, prefix="", evaluate='dev'):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if evaluate == 'dev':
            eval_dataset = cache_examples(args, eval_task, tokenizer, lexicon=args.lexicon, task_vocab=args.task_vocab, phrase='dev')
        else:
            eval_dataset = cache_examples(args, eval_task, tokenizer, lexicon=args.lexicon, task_vocab=args.task_vocab, phrase='test')
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)
        eval_steps = 0
        out_label_ids, preds = None, None
        CE_loss = 0.0
        for batch in eval_dataloader:
            model.eval() # 设置测试
            batch = tuple(t.to(args.device) for t in batch)
            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids": batch[2], "labels": batch[3],
                          'masked_input_ids': batch[-5], 'masked_attention_mask': batch[-4], 'candidate_words': batch[-3],
                          'masked_lm_labels': batch[-2], 'word_polarity': batch[-1]}
                outputs = model(**inputs)  # loss, logits, sequence_output, pooled_output
                loss_1, loss_lexicon, logits = outputs
                CE_loss += loss_1.item()
            eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)
        preds = np.argmax(preds, axis=1)
        result = compute_metrics('sst-2', preds, out_label_ids)
        results.update(result)
        return CE_loss/eval_steps, results['acc']


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", type=str, required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parser.add_argument("--model_type", type=str, required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--model_name_or_path", type=str, required=True,
        help="Path to pre-trained model")
    parser.add_argument("--task_name",type=str, required=True,
        help="Evaluation language. Also train language if `train_language` is set to None.",)

    # Other parameters
    parser.add_argument("--config_name", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",type=str, help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--cache_dir", type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",)
    parser.add_argument( "--max_seq_length", default=128, type=int, required=True,
        help="The maximum total input sequence length after tokenization. Sequences longer "
             "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, required=True, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=1, help="random seed for initialization")
    parser.add_argument("--fp16", action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument( "--fp16_opt_level", type=str,  default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
             "See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument( "--overwrite_output_dir", action="store_true", help="save models and cofigs")
    parser.add_argument("--dropout_prob", default=0.1, type=float)
    parser.add_argument("--lexicon_file", default="", type=str)
    parser.add_argument("--all_data_file", default="", type=str)

    parser.add_argument("--loss_type", default="aggregation", type=str, required=True) # aggregation, joint
    parser.add_argument("--loss_balance_type", default="weight_sum", type=str, required=True) # weight_sum, add_vec
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--a", default=0.5, type=float)
    parser.add_argument("--b", default=0.5, type=float)
    parser.add_argument("--c", default=1.0, type=float)
    parser.add_argument("--num_pt_epochs", default=1.0, type=float, help="Total number of training epochs to perform.")

    args = parser.parse_args()

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
    return args


def main():
    args = get_args()

    set_seed(args)

    processor = Processor(args.task_name)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower() # bert
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
        do_basic_tokenize = False
    )
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    config.dropout_prob = args.dropout_prob
    config.a = args.a
    config.b = args.b
    config.c = args.c
    config.loss_type = args.loss_type
    config.loss_balance_type = args.loss_balance_type

    logger.info("Training/evaluation parameters %s", args)
    t1 = open(args.lexicon_file, encoding='utf-8').readline()
    lexicon = literal_eval(t1)
    args.lexicon = lexicon

    from data_utils import get_vocab
    task_vocab = get_vocab(args.all_data_file, language='en')
    config.task_vocab = task_vocab
    args.task_vocab = task_vocab

    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)

    # Training
    if args.do_train:
        train_dataset = cache_examples(args, args.task_name, tokenizer, lexicon=lexicon, task_vocab=task_vocab, phrase='train') # here must False
        train(args, train_dataset, model, tokenizer)
    if not args.do_train and args.do_eval:
        logger.info("===========> inference ... ")
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            evaluate(args, model, tokenizer, evaluate='test')


if __name__ == "__main__":
    main()
