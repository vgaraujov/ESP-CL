from transformers import AdamW, get_linear_schedule_with_warmup
from torch import optim
from torch.utils.data import DataLoader
import argparse
import copy
import logging
import numpy as np
import os
import pickle
import torch
logger = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.WARNING)

from settings import parse_test_args, model_classes, init_logging
from utils import TextClassificationDataset, dynamic_collate_fn, prepare_inputs, DynamicBatchSampler


def test_task(task_id, args, model, test_dataset):

    if not args.no_fp16_test:
        model = model.half()

    def update_metrics(loss, logits, cur_loss, cur_acc):
        preds = np.argmax(logits, axis=1)
        return cur_loss + loss, cur_acc + np.sum(preds == labels.detach().cpu().numpy())

    cur_loss, cur_acc = 0, 0
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.n_workers,
                          shuffle=args.reproduce, collate_fn=dynamic_collate_fn)
    tot_n_inputs = 0
    for step, batch in enumerate(test_dataloader):
        n_inputs, input_ids, masks, labels = prepare_inputs(batch)
        tot_n_inputs += n_inputs
        with torch.no_grad():
            model.eval()
            outputs = model(input_ids=input_ids, attention_mask=masks, labels=labels)
            loss = outputs[0].item()
            logits = outputs[1].detach().cpu().numpy()
        cur_loss, cur_acc = update_metrics(loss*n_inputs, logits, cur_loss, cur_acc)
        if (step+1) % args.logging_steps == 0:
            logging.info("Tested {}/{} examples , test loss: {:.3f} , test acc: {:.3f}".format(
                tot_n_inputs, len(test_dataset), cur_loss/tot_n_inputs, cur_acc/tot_n_inputs))
    assert tot_n_inputs == len(test_dataset)

    logger.info("test loss: {:.3f} , test acc: {:.3f}".format(
        cur_loss / len(test_dataset), cur_acc / len(test_dataset)))

    return cur_acc / len(test_dataset)


def main():
    args = parse_test_args()
    train_args = pickle.load(open(os.path.join(args.output_dir, 'train_args'), 'rb'))
    assert train_args.output_dir == args.output_dir
    args.__dict__.update(train_args.__dict__)
    init_logging(os.path.join(args.output_dir, 'log_test.txt'))
    logger.info("args: " + str(args))

    config_class, model_class, args.tokenizer_class = model_classes[args.model_type]
    model_config = config_class.from_pretrained(args.model_name, num_labels=args.n_labels, hidden_dropout_prob=0, attention_probs_dropout_prob=0)

    model_config.highway_pen = args.highway_pen

    save_model_path = os.path.join(args.output_dir, 'checkpoint-{}'.format(len(args.tasks)-1))
    model = model_class.from_pretrained(save_model_path, config=model_config).cuda()

    avg_acc = 0
    for task_id, task in enumerate(args.tasks):
        logger.info("Start testing {}...".format(task))
        test_dataset = pickle.load(open(os.path.join(args.output_dir, 'test_dataset-{}'.format(task_id)), 'rb'))
        task_acc = test_task(task_id, args, model, test_dataset)
        avg_acc += task_acc / len(args.tasks)
    logger.info("Average acc: {:.3f}".format(avg_acc))


if __name__ == "__main__":
    main()
