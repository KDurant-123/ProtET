# ProtET: Multi-Modal CLIP-Informed Protein Editing
The repository is an official implementation of [Multi-Modal CLIP-Informed Protein Editing](https://arxiv.org/abs/2407.19296)
<img src="Figures/protein-biotext.png" alt="data" width="800" height="600"> \

## Abstract
**Background:** Proteins govern most biological functions essential for life, and achieving controllable protein editing has made great advances in probing natural systems, creating therapeutic conjugates and generating novel protein constructs. Recently, machine learning-assisted protein editing (MLPE) has shown promise in accelerating optimization cycles and reducing experimental workloads. However, current methods struggle with the vast combinatorial space of potential protein edits and cannot explicitly conduct protein editing using biotext instructions, limiting their interactivity with human feedback. **Methods:** To fill these gaps, we propose a novel method called ProtET for efficient CLIP-informed protein editing through multi-modality learning. Our approach comprises two stages: in the pretraining stage, contrastive learning aligns protein-biotext representations encoded by two large language models (LLMs), respectively. Subsequently, during the protein editing stage, the fused features from editing instruction texts and original protein sequences serve as the final editing condition for generating target protein sequences. **Results:** Comprehensive experiments demonstrated the superiority of ProtET in editing proteins to enhance human-expected functionality across multiple attribute domains, including enzyme catalytic activity, protein stability and antibody specific binding ability. And ProtET improves the state-of-the-art results by a large margin, leading to significant stability improvements of 16.67% and 16.90%. **Conclusions:** This capability positions ProtET to advance real-world artificial protein editing, potentially addressing unmet academic, industrial, and clinical needs.

## Overview
ProtET is a multi-modality deep learning model that hybridly encodes biological languages and natural languages, and then executes cross-modal generation to achieve controllable protein editing. 
To accomplish this, we first curate millions of protein-biotext aligned pairs, each comprising protein sequences and functional biotext annotations, as illustrated in Figure 1. The large-scale multi-modal dataset consists of 570,420 proteins with manually reviewed property annotations and 251,131,639 proteins with computationally analyzed annotations. We then construct transformer-structured encoder-based models (*i.e.*, a large protein model with 650 million trainable parameters and a large language model with 100 million trainable parameters) to encode the features of both protein sequences and biotexts, respectively. Additionally, a hierarchical training paradigm is proposed to alleviate the challenge of cross-modal protein editing. During the pretraining stage, similar to CLIP, our multi-modality pretraining is performed using contrastive learning objectives to align the features of the protein and biotext, facilitating easier editing instruction. In the editing stage, the aligned protein features and desired function description features extracted by the pretrained models are fused by the introduced FiLM module. And we construct a generative decoder model to design the desired protein sequences in an auto-regressive manner. ProtET innovatively introduces a novel protein editing paradigm through multi-modal pretraining and cross-modal generation. Its controllable protein editing capability to enhance human-expected functionality demonstrates the great potential for clinical applications, such as vaccine development and genetic therapy, etc.
<img src="Figures/framework.png" alt="framework" width="800" height="600"> \
Figure 1: Overview of ProtET framework.

## Implement details
The overall framework are trained with a batch size of 128 for 10 epochs, utilizing 16 NVIDIA 32G V100 GPUs. The learning rate is initialized as $5.0 \times 10^{-5}$ with 2,000 linear warm-up steps.

## Environment installation
### Create a virtual environment
```
conda create -n ProtET python=3.10
conda activate ProtET
```
### Install Packages
```
pip install -r requirements.txt
```

## Model Checkpoint
[ProtET]()

### Load the multi-modality aligned PLM


## downstream tasks


## Citation
If you find this repository useful, please cite our paper:
```
@article{yin2024multi,
  title={Multi-Modal Clip-Informed Protein Editing},
  author={Yin, Mingze and Zhou, Hanjing and Zhu, Yiheng and Lin, Miao and Wu, Yixuan and Wu, Jialu and Xu, Hongxia and Hsieh, Chang-Yu and Hou, Tingjun and Chen, Jintai and Wu, Jian},
  journal={Health Data Science},
  year={2024},
  publisher={AAAS}
}
```


