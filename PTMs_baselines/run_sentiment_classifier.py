import os
import glob
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from load_data import cache_examples_train, cache_examples_dev, cache_examples_test

from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification
from transformers.models.xlnet.modeling_xlnet import XLNetForSequenceClassification

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
from data_utils import output_modes
from data_utils import processors


logger = logging.getLogger(__name__)
MODEL_CLASSES = {
    "bert_raw": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta_raw": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "roberta_large_skep": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "xlnet_raw": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),

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

    # Prepare optimizer and schedule (linear warmup and decay)
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

    # Check if saved optimizer or scheduler states exist
    # if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
    #     os.path.join(args.model_name_or_path, "scheduler.pt")
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

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
    steps_trained_in_current_epoch = 0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    best_dev_acc = 0.0
    for epoch in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = [t.to(args.device) for t in batch]
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[-1]}
            inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert"] else None)
            outputs = model(**inputs) # loss, logits, sequence_output, pooled_output
            loss, logits = outputs[:2]  # model outputs are always tuple in transformers (see doc)
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
        results = evaluate(args, model, tokenizer, evaluate='dev')
        if results['acc'] >= best_dev_acc:
            best_dev_acc = results['acc']
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
                logger.info("\n===========> epoch: %s, best dev to test, dev acc = %s", (epoch+1), best_dev_acc)
                evaluate(args, model, tokenizer, evaluate='test')
            else:
                logger.info("\n==============================> epoch: %s, best dev is = %s", (epoch + 1), best_dev_acc)


def evaluate(args, model, tokenizer, prefix="", evaluate='dev'):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        if evaluate == 'dev':
            eval_dataset = cache_examples_dev(args, eval_task, tokenizer)
        else:
            eval_dataset = cache_examples_test(args, eval_task, tokenizer)
        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in eval_dataloader:
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[-1]}
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert"] else None)  # XLM and DistilBERT don't use segment_ids
                outputs = model(**inputs)
                loss, logits = outputs[:2]

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
        if evaluate == 'dev':
            logger.info("***** Eval results {} *****".format(prefix))
        else:
            logger.info("***** Test results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
    return results


def main():
    from args import get_args
    args = get_args()

    # Set seed
    set_seed(args)

    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    # https://stackoverflow.com/questions/59760328/how-does-torch-distributed-barrier-work
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
    if args.model_type == 'xlnet' or args.model_type == 'xlnet_tracenet':
        config.d_model = config.hidden_size
    # else:
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)

    # Training

    if args.do_train:
        train_dataset = cache_examples_train(args, args.task_name, tokenizer) # here must False
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
