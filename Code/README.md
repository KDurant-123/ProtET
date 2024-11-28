# Source codes

本代码设置为deepspeed zero1, gradient_checkpointing=True，预训练和下游常用bsz=64

## DLC shell

pip uninstall -y torchaudio torchdata torchvision torch
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
pip install torchdrug
pip install scikit-learn pandas decorator ipython networkx tqdm matplotlib
pip install fair-esm easydict pyyaml lmdb
pip install deepspeed==0.12.4
pip install transformers==4.35.2 accelerate==0.25.0 datasets peft
pip install wandb
pip install hydra-core
pip uninstall -y transformer-engine
pip uninstall -y apex
cd /root/code/protein/
sh FST/train_pai.sh  或者  sh FST/downstream_pai.sh

## 各种路径

数据采样的代码路径: sampling.py

预训练数据集路径: oss://citybrain-ai4s/zhouhanjing/datasets/ProtSTData/ProtDescribe/<name>.yaml

                其中train.json是原来0.5M的训练数据，train_new.json是新采样后的数据，融合了251M+0.5M
                
                要修改预训练数据集，需要在pretraining_task.py 190行改

下游数据集路径: oss://citybrain-ai4s/zhouhanjing/datasets/ProtSTData/<dataset_name>

下游数据集config路径: config/downstream/<dataset_name>.yaml

最终ckpt路径: oss://citybrain-ai4s/zhouhanjing/FST/newdata_mlm0.7_sparc/checkpoint-22200

            只需要用到该目录下的config.json和model.safetensors

## 预训练的sh示例(用上所有loss)

export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export WANDB_PROJECT="ProtST_pretrain"

echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l) # use all_available gpus
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "worker-gpu: $(nvidia-smi -L | wc -l)"

echo "================================ddp_options===================================="
ddp_options="--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$n_gpu --nnodes=$WORLD_SIZE --node_rank=$RANK"
echo "ddp_options: ${ddp_options}"
echo "==============================================================================="

torchrun $ddp_options /root/code/protein/FST/pretrain.py \
--deepspeed --deepspeed_config /root/code/protein/FST/zero1.json \
--batch_size 64 \
--t2p_mlm 1 \
--local_contrast 1 \
--output_path /root/data/FST/newdata_mlm0.7_sparc

## 下游的sh示例(用mlm0.7_5epoch的ckpt在GPMF数据集上跑，full-finetune)

export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL
export WANDB_PROJECT="ProtST_downstream"

echo "==================================MP==========================================="
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l) # use all_available gpus
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo "NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME}"
echo "WORLD_SIZE: ${WORLD_SIZE}"
echo "worker-gpu: $(nvidia-smi -L | wc -l)"

echo "================================ddp_options===================================="
ddp_options="--master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$n_gpu --nnodes=$WORLD_SIZE --node_rank=$RANK"
echo "ddp_options: ${ddp_options}"
echo "==============================================================================="

torchrun $ddp_options /root/code/protein/FST/downstream.py \
--deepspeed --deepspeed_config /root/code/protein/FST/zero1.json \
--output_path /root/data/FST/GOMF/ff_mlm0.7_5 \
--model_name /root/data/FST/mlm0.7/checkpoint-5280 \
--task MultiLabelSequenceClassificationTask \
--dataset GeneOntology_MF \
--num_labels 489 \
--metric_for_best_model f1_max 