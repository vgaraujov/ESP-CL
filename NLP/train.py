from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import logging
import numpy as np
import os
import pickle
import torch
import random
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)

from settings import parse_train_args, model_classes, init_logging
from utils import TextClassificationDataset, DynamicBatchSampler
from utils import dynamic_collate_fn, prepare_inputs


def compute_factor(entropies):
    entropy_s = torch.zeros(len(entropies))
    for i, highway_exit in enumerate(entropies):
        entropy = highway_exit[1]
        entropy_s[i] = entropy.mean()

    soft_entropy = torch.nn.functional.softmax(entropy_s, dim=0)*(len(entropies))
    soft_entropy[ soft_entropy > 1.0 ] = 1.0
    soft_entropy = 1-soft_entropy

    return soft_entropy

def multiply_entropy(soft_entropy, model):
    dict_entropy = { "layer.{}.".format(i): s for i, s in enumerate(soft_entropy) }

    for name,p in model.named_parameters():
        for k in dict_entropy.keys():
            if k in name:
                try:
                    p.grad *= dict_entropy[k]
                except:
                    continue
    
    return soft_entropy.cpu().detach()

def train_pipeline(model_config, task, task_id, args, train_dataset, memory_set, model):

    # Update Memory
    len_dataset = len(train_dataset)
    indices = random.sample(list(range(len_dataset)), int(len_dataset*args.mem_capacity))
    logger.info("Add {} elements from task {} to memory".format(len(indices), task))
    new_memory = torch.utils.data.Subset(train_dataset, indices)
    if memory_set is None:
        memory_set = new_memory
    else:
        memory_set = torch.utils.data.ConcatDataset((memory_set, new_memory))
    logger.info("Elements in Memory: {}".format(len(memory_set)))

    # Train Highways
    if model_config.highway_pen > 0:
        logger.info("Frozen BERT layer, Task: {}...".format(task))
        unfrozen_name = ["highway."+str(i)+"." for i in range(12)]
        for name, p in model.named_parameters():
            if sum([True if i in name else False for i in unfrozen_name]):
                p.requires_grad = True
            else:
                p.requires_grad = False

        logger.info("It is a Highway to training ...")
        # train only with replay memory
        train_task(args, model, memory_set, False, True, model_config.highway_pen, task_id)

    # Train BERT
    logger.info("UnFrozen BERT layer, Task: {}...".format(task))
    frozen_name = ["highway."+str(i)+"." for i in range(12)]
    for name, p in model.named_parameters():
        if sum([True if i in name else False for i in frozen_name]):
            p.requires_grad = False
        else:
            p.requires_grad = True

    if args.only_mem: # train whole model with only or all
        bert_dataset = memory_set
    else:
        bert_dataset = torch.utils.data.ConcatDataset((memory_set, train_dataset))

    logger.info("Start LLL training {}...".format(task))
    final_perct_task = train_task(args, model, bert_dataset, True, False, model_config.highway_pen, task_id)
    model_save_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(task_id))
    if task_id == len(args.tasks)-1:
        torch.save(model.state_dict(), model_save_path)
    if len(final_perct_task) > 0:
        torch.save({ 'entropy_task': final_perct_task }, os.path.join(args.output_dir, 'entropy_level_{}.pth'.format(task_id)))

    return memory_set

def train_task(args, model, train_dataset, train_bert, use_highway, highway_pen, task_id):
    
    total_n_dataset = len(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                                  shuffle=True, collate_fn=dynamic_collate_fn)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=len(train_dataset)//10)

    model.zero_grad()
    tot_epoch_loss, tot_n_inputs = 0, 0
    perc_entropy = []
    final_perct_task = []
    accum_iter = args.grad_accumulation_steps

    for step, batch in enumerate(train_dataloader):
        model.train()
        n_inputs, input_ids, masks, labels = prepare_inputs(batch)
        output = model(input_ids=input_ids, attention_mask=masks, labels=labels, train_highway=use_highway)
        loss = output[0] / accum_iter

        if train_bert:
            # Dynamic Freezing
            factors = compute_factor(output[-1])
            if args.dynamic_freeze:
                bool_factor = (factors > 0).type(torch.uint8)
                frozen_name = []
                for i, val in enumerate(bool_factor):
                    if val == 0:
                        frozen_name.append("layer."+str(i)+".")
                for name, p in model.named_parameters():
                    if sum([True if i in name else False for i in frozen_name]):
                        p.requires_grad = False
                    elif "highway" in name:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if train_bert and highway_pen > 0:
            perc_entropy.append(multiply_entropy(factors, model))
        if ((step + 1) % accum_iter == 0) or (step + 1 == len(train_dataloader)):
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        
        tot_n_inputs += n_inputs
        tot_epoch_loss += loss.item() * n_inputs

        if (step+1) % args.logging_steps == 0:
            logger.info("progress: {:.2f} , step: {} , lr: {:.2E} , avg batch size: {:.1f} , avg loss: {:.3f}".format(
                tot_n_inputs/total_n_dataset, step+1, scheduler.get_lr()[0], tot_n_inputs//(step+1), tot_epoch_loss/tot_n_inputs))

            if len(perc_entropy) > 0:
                perc_entropy = torch.stack(perc_entropy)
                logger.info(perc_entropy.mean(dim=0))
                final_perct_task.append(perc_entropy.mean(dim=0))
                perc_entropy = []
        torch.cuda.empty_cache() # this saves memory but slows down the process

    logger.info("Finish training, avg loss: {:.3f}".format(tot_epoch_loss/tot_n_inputs))
    del optimizer, optimizer_grouped_parameters, train_dataloader

    return final_perct_task


def main():
    args = parse_train_args()
    pickle.dump(args, open(os.path.join(args.output_dir, 'train_args'), 'wb'))
    init_logging(os.path.join(args.output_dir, 'log_train.txt'))
    logger.info("args: " + str(args))

    logger.info("Initializing main {} model".format(args.model_name))
    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    tokenizer = args.tokenizer_class.from_pretrained(args.model_name)

    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels)
    model_config.highway_pen = args.highway_pen
    config_save_path = os.path.join(args.output_dir, 'config')
    model_config.to_json_file(config_save_path)
    model = model_class.from_pretrained(args.model_name, config=model_config).cuda()
    memory = None

    for task_id, task in enumerate(args.tasks):
        logger.info("Start parsing {} train data...".format(task))
        train_dataset = TextClassificationDataset(task, "train", args, tokenizer)
        logger.info("Start training {}...".format(task))
        logger.info("Len dataset {}...".format(len(train_dataset)))
        memory = train_pipeline(model_config, task, task_id, args, train_dataset, memory, model)
        model_save_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(task_id))
        if task_id == len(args.tasks)-1:
            torch.save(model.state_dict(), model_save_path)

    for task_id, task in enumerate(args.tasks):
        logger.info("Start parsing {} test data...".format(task))
        test_dataset = TextClassificationDataset(task, "test", args, tokenizer)
        pickle.dump(test_dataset, open(os.path.join(args.output_dir, 'test_dataset-{}'.format(task_id)), 'wb'))

if __name__ == "__main__":
    main()
