import os
from model import MSD_model
from train import train
from Dataset_Load import MyDataset
import torch
import argparse
import random
import numpy as np
from transformers import CLIPProcessor
import pickle
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='2', type=str, help='device number')
    parser.add_argument('--model', default='MSD_model', type=str, help='the model name', choices=['MSD_model'])
    parser.add_argument('--text_name', default='text_json_final', type=str, help='the text data folder name')
    parser.add_argument('--simple_linear', default=False, type=bool, help='linear implementation choice')
    parser.add_argument('--num_train_epochs', default=10, type=int, help='number of train epoched')
    parser.add_argument('--train_batch_size', default=128, type=int, help='batch size in train phase')
    parser.add_argument('--dev_batch_size', default=128, type=int, help='batch size in dev phase')
    parser.add_argument('--label_number', default=2, type=int, help='the number of classification labels')
    parser.add_argument('--text_size', default=512, type=int, help='text hidden size')
    parser.add_argument('--image_size', default=768, type=int, help='image hidden size')
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--optimizer_name", type=str, default='adam',
                        help="use which optimizer to train the model.")
    parser.add_argument('--learning_rate', default=5e-4, type=float, )
    parser.add_argument('--clip_learning_rate', default=1e-6, type=float, )
    parser.add_argument('--max_len', default=77, type=int, )
    parser.add_argument('--layers', default=3, type=int, help='number of transform layers')
    parser.add_argument('--max_grad_norm', default=5.0, type=float, help='grad clip norm')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='weight decay')
    parser.add_argument('--warmup_proportion', default=0.2, type=float, help='warm up proportion')
    parser.add_argument('--dropout_rate', default=0.1, type=float, help='dropout rate')
    parser.add_argument('--output_dir', default='MMSD2.0-main/output_dir/', type=str, help='the output path')
    parser.add_argument('--limit', default=None, type=int, help='the limited number of training examples')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    return parser.parse_args()


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = set_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    device = torch.device("cuda:2" if torch.cuda.is_available() and int(args.device) >= 0 else "cpu")

    seed_everything(args.seed)


    train_data = MyDataset(mode='train', text_name=args.text_name, limit=None)
    dev_data = MyDataset(mode='valid', text_name=args.text_name, limit=None)
    test_data = MyDataset(mode='test', text_name=args.text_name, limit=None)
    if args.model == 'MSD_model':
        bertl_text = RobertaModel.from_pretrained(opt["roberta_path"])
        model = MSD_model(args)
    else:
        raise RuntimeError('Error model name!')

    model.to(device)

    train(args, model, device, train_data, dev_data, test_data, processor)



if __name__ == '__main__':
    main()
