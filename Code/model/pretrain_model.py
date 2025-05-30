import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import get_rank, get_world_size
from torch.distributed.nn.functional import all_gather
from torch.nn.functional import cross_entropy
from transformers import PreTrainedModel, PretrainedConfig, AutoConfig, AutoModel

from model.layers import CrossAttention, CLIPLoss
from model.bank import AttrProtoBank
from model.attention import LocalCrossAttention, FragmentDecoder
from model.modeling_esm import MyEsmAutoModel
import pdb

class ProteinTextCLIPConfig(PretrainedConfig):
    model_type = "protein_text_clip"
    is_composition = True

    def __init__(self,
                 protein_model_config,
                 text_model_config,
                 projection_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.protein_model_config = protein_model_config
        self.text_model_config = text_model_config

        if isinstance(protein_model_config, dict):
            self.protein_model_config = AutoConfig.for_model(**protein_model_config)
        if isinstance(text_model_config, dict):
            self.text_model_config = AutoConfig.for_model(**text_model_config)
        self.projection_dim = projection_dim

        self.hidden_sizes = [self.protein_model_config.hidden_size,
                             self.text_model_config.hidden_size,
                             self.projection_dim]
        self.logit_scale_init = kwargs.pop("logit_scale_init", 0.07)


class ProteinTextCLIPForPretrain(PreTrainedModel):
    config_class = ProteinTextCLIPConfig

    def __init__(self, config):
        super().__init__(config)
        protein_model_config = config.protein_model_config
        text_model_config = config.text_model_config

        self.protein_model = AutoModel.from_pretrained(protein_model_config._name_or_path)
        self.text_model = AutoModel.from_pretrained(text_model_config._name_or_path)

        self.protein_projection = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.config.logit_scale_init))

    def forward(self, protein_input_ids, protein_attention_mask, text_input_ids, text_attention_mask):
        protein_embeds = self.protein_model(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask
        ).last_hidden_state.mean(dim=1)
        protein_embeds = self.protein_projection(protein_embeds)

        text_embeds = self.text_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state.mean(dim=1)
        text_embeds = self.text_projection(text_embeds)

        # normalize the embeddings
        protein_embeds = protein_embeds / protein_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)

        cl_loss = CLIPLoss(
            local_loss=False,
            gather_with_grad=True,
            cache_labels=True,
            rank=get_rank(),
            world_size=get_world_size()
        )(protein_embeds, text_embeds, self.logit_scale.exp())

        return {
            "loss": cl_loss,
            "cl_loss": cl_loss,
            "logit_scale": self.logit_scale.exp()
        }


class ETConfig(PretrainedConfig):
    model_type = "ET"
    is_composition = True

    def __init__(self,
                 protein_model_config,
                 text_model_config,
                 mlp_num_layers,
                 fusion_num_heads,
                 projection_dim,
                 fusion_num_layers,
                 fusion_batch_norm,
                 **kwargs):
        super().__init__(**kwargs)

        self.protein_model_config = protein_model_config
        self.text_model_config = text_model_config

        if isinstance(protein_model_config, dict):
            self.protein_model_config = AutoConfig.for_model(**protein_model_config)
        if isinstance(text_model_config, dict):
            self.text_model_config = AutoConfig.for_model(**text_model_config)
        self.projection_dim = projection_dim

        self.mlp_num_layers = mlp_num_layers
        self.fusion_num_heads = fusion_num_heads
        self.projection_dim = projection_dim
        self.fusion_num_layers = fusion_num_layers
        self.fusion_batch_norm = fusion_batch_norm
        self.hidden_sizes = [self.protein_model_config.hidden_size,
                             self.text_model_config.hidden_size,
                             self.projection_dim]
        self.logit_scale_init = kwargs.pop("logit_scale_init", 0.07)
        self.protein_mask_probability = kwargs.pop("protein_mask_probability", 0.15)
        self.text_mask_probability = kwargs.pop("text_mask_probability", 0.15)


class ETForPretrain(PreTrainedModel):
    # enable gradient_checkpointing
    _keys_to_ignore_on_load_missing = [r"position_ids", "lm_head.decoder.weight"]
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    supports_gradient_checkpointing = True

    config_class = ETConfig

    def __init__(self, config):
        super().__init__(config)
        protein_model_config = config.protein_model_config
        text_model_config = config.text_model_config

        # enable gradient_checkpointing
        self.protein_model = MyEsmAutoModel.from_pretrained(protein_model_config._name_or_path)
        
        self.text_model = AutoModel.from_pretrained(text_model_config._name_or_path)
        self.protein_projection = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )
        self.text_projection = nn.Sequential(
            nn.Linear(text_model_config.hidden_size, self.config.projection_dim),
            nn.ReLU(),
            nn.Linear(self.config.projection_dim, self.config.projection_dim),
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.config.logit_scale_init))

        self.mlm_head = nn.Sequential(
            nn.Linear(protein_model_config.hidden_size, self.config.projection_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.projection_dim),
            nn.Linear(self.config.projection_dim, protein_model_config.vocab_size),
        )
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self,
                protein_input_ids,
                protein_attention_mask,
                text_input_ids,
                text_attention_mask,
                protein_masked_input_ids,
                protein_masked_labels
                ):

        text_outputs = self.text_model(
            input_ids=text_input_ids, attention_mask=text_attention_mask
        ).last_hidden_state
        text_embeds = text_outputs.mean(dim=1) #[bsz, dim]
        text_embeds = self.text_projection(text_embeds)

        protein_outputs = self.protein_model(
            input_ids=protein_input_ids, attention_mask=protein_attention_mask
        ).last_hidden_state
        protein_embeds = protein_outputs.mean(dim=1) #[bsz, dim]
        protein_embeds = self.protein_projection(protein_embeds)

        # normalize the embeddings
        protein_embeds = protein_embeds / protein_embeds.norm(dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        loss_dict={}
        cl_loss = CLIPLoss(
            local_loss=False,
            gather_with_grad=True,
            cache_labels=True,
            rank=get_rank(),
            world_size=get_world_size()
        )(protein_embeds, text_embeds, self.logit_scale.exp())
        loss_dict['cl_loss'] = cl_loss
        loss_dict['loss'] = cl_loss
        del text_embeds
        del protein_embeds

        # compute outputs
        protein_outputs1 = self.protein_model(input_ids=protein_masked_input_ids,
                                             attention_mask=protein_attention_mask).last_hidden_state
        # compute the mlm loss
        protein_mlm_logits = self.mlm_head(protein_outputs1)
        protein_mlm_loss = cross_entropy(protein_mlm_logits.view(-1, protein_mlm_logits.shape[-1]),
                                         protein_masked_labels.view(-1))
        loss_dict['protein_mlm_loss'] = protein_mlm_loss
        loss_dict['loss'] += protein_mlm_loss
        loss_dict['logit_scale'] = self.logit_scale.exp()
        return loss_dict