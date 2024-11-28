import hydra
import transformers
import wandb
import os
from model.pretrain_model import ETForPretrain
import deepspeed
import argparse
import yaml

wandb.init(mode='disabled')

transformers.logging.set_verbosity_info()
pretrain_tasks = __import__('task.pretrain_task', fromlist='*')

def main():
    parser = argparse.ArgumentParser(description='ProtET')
    parser.add_argument('--task_name', type=str, default='esm2_t33_650M_UR50D_ProtET')
    # path
    parser.add_argument('--data_path', type=str, default='/root/data/datasets')
    parser.add_argument('--output_path', type=str, default='/root/data/ET/')
    # model
    parser.add_argument('--protein_model_name', type=str, default='/root/data/backbones/esm2_t33_650M_UR50D')
    parser.add_argument('--protein_model_fixed', type=bool, default=False)
    parser.add_argument('--text_model_name', type=str, default='/root/data/backbones/BiomedNLP-PubMedBERT-base-uncased-abstract')
    parser.add_argument('--text_model_fixed', type=bool, default=True)
    parser.add_argument('--projection_dim', type=int, default=512)
    parser.add_argument('--fusion_num_heads', type=int, default=8)
    parser.add_argument('--fusion_num_layers', type=int, default=1)
    parser.add_argument('--fusion_batch_norm', type=bool, default=True)
    parser.add_argument('--mlp_num_layers', type=int, default=2)
    parser.add_argument('--mlm_probability', type=float, default=0.15)    
    # dataset
    parser.add_argument('--dataset', type=str, default='ProtETData')
    parser.add_argument('--max_length', type=int, default=1024)
    # train
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--lr_ratio', type=float, default=0.1)
    parser.add_argument('--fp16', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=35)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    # eval
    parser.add_argument('--metric_for_best_model', type=str, default='loss')
    # task
    parser.add_argument('--task', type=str, default='ETPretrainTask')
    # RANK
    parser.add_argument('--local_rank', type=int, default=0)
    # deepspeed
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    task = getattr(pretrain_tasks, args.task)(args)
    task.run()


if __name__ == '__main__':
    main()
