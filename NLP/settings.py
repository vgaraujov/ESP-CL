from transformers import BertTokenizer, BertConfig
from modeling_bert import BertForSequenceClassification
import argparse
import datetime
import logging
import os
import shutil
import torch
import random
import numpy as np

model_classes = {
    'bert': (BertConfig, BertForSequenceClassification, BertTokenizer),
}
label_offsets = {
    'ag_news_csv': -1,
    'amazon_review_full_csv': 3,
    'dbpedia_csv': 8,
    'yahoo_answers_csv': 22,
    'yelp_review_full_csv': 3
}


def set_device_id(args):
    args.device_id = torch.cuda.current_device() # not for multi-gpu
    torch.cuda.set_device(args.device_id)

def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # gpu vars
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

def parse_train_args():
    parser = argparse.ArgumentParser("Train Lifelong Language Learning")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--logging_steps", type=int, default=200)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--model_type", type=str, default="bert", help="Model type selected in the list: " + ", ".join(model_classes.keys()))
    parser.add_argument("--n_labels", type=int, default=33)
    parser.add_argument("--n_test", type=int, default=7600)
    parser.add_argument("--n_train", type=int, default=115000)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--valid_ratio", type=float, default=0)
    parser.add_argument("--output_dir", type=str, default="output0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--reproduce", action="store_true")
    parser.add_argument("--tasks", nargs='+', default=["ag_news_csv"])
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--highway_pen", type=float, default=0.5)
    parser.add_argument("--mem_capacity", type=float, default=0.2)
    parser.add_argument("--only_mem", action="store_true")
    parser.add_argument("--dynamic_freeze", action="store_true")

    args = parser.parse_args()

    if args.debug:
        args.n_train = 500
        args.logging_steps = 1
        args.n_test = 100
        args.output_dir = "output_debug"
        args.overwrite = True

    set_device_id(args)
    set_seed(args)

    if os.path.exists(args.output_dir):
        if args.overwrite:
            choice = 'y'
        else:
            choice = input("Output directory ({}) exists! Remove? ".format(args.output_dir))
        if choice.lower()[0] == 'y':
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            raise ValueError("Output directory exists!")
    else:
        os.makedirs(args.output_dir)

    return args


def parse_test_args():
    parser = argparse.ArgumentParser("Test Lifelong Language Learning")
    parser.add_argument("--no_fp16_test", action="store_true")
    parser.add_argument("--output_dir", type=str, default="output0")

    args = parser.parse_args()
    set_device_id(args)
    return args


class TimeFilter(logging.Filter):
    def filter(self, record):
        try:
            last = self.last
        except AttributeError:
            last = record.relativeCreated

        delta = record.relativeCreated/1000 - last/1000
        record.relative = "{:.3f}".format(delta)
        record.uptime = str(datetime.timedelta(seconds=record.relativeCreated//1000))
        self.last = record.relativeCreated
        return True


def init_logging(filename):
    logging_format = "%(asctime)s - %(uptime)s - %(relative)ss - %(levelname)s - %(name)s - %(message)s"
    logging.basicConfig(format=logging_format, filename=filename, filemode='a', level=logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(logging_format))
    root_logger = logging.getLogger()
    root_logger.addHandler(console_handler)
    for handler in root_logger.handlers:
        handler.addFilter(TimeFilter())
