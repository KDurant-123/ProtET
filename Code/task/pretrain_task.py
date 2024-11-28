from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback, AutoConfig, AutoTokenizer
import numpy as np
from model.pretrain_model import ProteinTextCLIPForPretrain, ETForPretrain, ProteinTextCLIPConfig, ETConfig
from trainer.pretrain_trainer import CLIPPretrainTrainer
from utils import DataCollatorForProteinTextCLIPPretrain, DataCollatorForETPretrain


class PretrainTask(object):
    def __init__(self, run_config):
        self.run_config = run_config
        self.task_model = self.build_task_model()
        self.dataset = self.build_dataset()
        self.train_args = self.build_train_args()
        self.trainer = self.build_trainer()

    def build_task_model(self):
        raise NotImplementedError()

    def build_dataset(self):
        raise NotImplementedError()

    def build_train_args(self):
        raise NotImplementedError()

    def build_trainer(self):
        raise NotImplementedError()

    def run(self):
        self.trainer.train()


class ProteinTextCLIPPretrainTask(PretrainTask):
    def __init__(self, run_config):
        self.protein_model_config = AutoConfig.from_pretrained(run_config.protein_model_name)
        self.text_model_config = AutoConfig.from_pretrained(run_config.text_model_name)
        self.protein_tokenizer = AutoTokenizer.from_pretrained(run_config.protein_model_name)
        self.text_tokenizer = AutoTokenizer.from_pretrained(run_config.text_model_name)
        super().__init__(run_config)

    def build_task_model(self):
        task_model_config = ProteinTextCLIPConfig(
            protein_model_config=self.protein_model_config,
            text_model_config=self.text_model_config,
            projection_dim=self.run_config.projection_dim,
        )
        return ProteinTextCLIPForPretrain(task_model_config)

    def build_dataset(self):
        def preprocess_function(examples):
            protein_tokenized_examples = self.protein_tokenizer(examples["seq"],
                                                                max_length=self.run_config.max_length,
                                                                truncation=True, padding=False)
            text_tokenized_examples = self.text_tokenizer(examples["text"], max_length=512,
                                                          truncation=True, padding=False)
            return {
                'protein_input_ids': protein_tokenized_examples['input_ids'],
                'protein_attention_mask': protein_tokenized_examples['attention_mask'],
                'text_input_ids': text_tokenized_examples['input_ids'],
                'text_attention_mask': text_tokenized_examples['attention_mask'],
            }

        dataset = load_dataset("json", data_files={
            'train': f'{self.run_config.data_path}/{self.run_config.dataset}/train.json',
            'valid': f'{self.run_config.data_path}/{self.run_config.dataset}/valid.json',
        })
        dataset = dataset.map(preprocess_function, batched=True, num_proc=8)
        dataset.set_format(type='torch', columns=['protein_input_ids', 'protein_attention_mask',
                                                  'text_input_ids', 'text_attention_mask'])
        return dataset

    def build_train_args(self):
        return TrainingArguments(
            output_dir=self.run_config.output_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=50000,
            per_device_train_batch_size=self.run_config.batch_size,
            per_device_eval_batch_size=self.run_config.batch_size,
            num_train_epochs=self.run_config.num_epochs,
            weight_decay=self.run_config.weight_decay,
            fp16=self.run_config.fp16,
            push_to_hub=False,
            learning_rate=self.run_config.lr,
            report_to=["wandb"],
            warmup_ratio=self.run_config.warmup_ratio,
            load_best_model_at_end=True,
            label_names=["protein_input_ids"]  # hack fix for 'eval_loss' not found error
        )

    def build_trainer(self):
        return CLIPPretrainTrainer(
            model=self.task_model,
            args=self.train_args,
            data_collator=DataCollatorForProteinTextCLIPPretrain(self.protein_tokenizer,
                                                                 self.text_tokenizer,
                                                                 mlm_probability=getattr(self.run_config,
                                                                                         "mlm_probability", 0.0)),
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
            protein_model_fixed=self.run_config.protein_model_fixed,
            text_model_fixed=self.run_config.text_model_fixed,
            lr_ratio=self.run_config.lr_ratio,
        )


class ETPretrainTask(ProteinTextCLIPPretrainTask):
    def build_dataset(self):
        def preprocess_function(examples):
            protein_tokenized_examples = self.protein_tokenizer(examples["seq"],
                                                                max_length=self.run_config.max_length,
                                                                truncation=True, padding=False)
            # add head and tail tokens to protein seq, and shrink some seqs to max_length
            text_tokenized_examples = self.text_tokenizer(examples["text"], max_length=512,
                                                          truncation=True, padding=False)
            
            return {
                'protein_input_ids': protein_tokenized_examples['input_ids'],
                'protein_attention_mask': protein_tokenized_examples['attention_mask'],
                'text_input_ids': text_tokenized_examples['input_ids'], #[B, token_len]
                'text_attention_mask': text_tokenized_examples['attention_mask'], #[B, token_len]
            }

        dataset = load_dataset("json", data_files={
            'train': f'{self.run_config.data_path}/{self.run_config.dataset}/train.json',
            'valid': f'{self.run_config.data_path}/{self.run_config.dataset}/valid.json',
        })
        dataset = dataset.map(preprocess_function, batched=True, num_proc=8)
        dataset.set_format(type='torch', columns=['protein_input_ids', 'protein_attention_mask',
                                                  'text_input_ids', 'text_attention_mask'])
        return dataset
    
    def build_task_model(self):
        task_model_config = ETConfig(
            protein_model_config=self.protein_model_config,
            text_model_config=self.text_model_config,
            projection_dim=self.run_config.projection_dim,
            mlp_num_layers=self.run_config.mlp_num_layers,
            fusion_num_heads=self.run_config.fusion_num_heads,
            fusion_num_layers=self.run_config.fusion_num_layers,
            fusion_batch_norm=self.run_config.fusion_batch_norm,
        )

        return ETForPretrain(task_model_config)
    
    def build_train_args(self):
        return TrainingArguments(
            output_dir=self.run_config.output_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="steps",
            logging_steps=10000,
            per_device_train_batch_size=self.run_config.batch_size,
            per_device_eval_batch_size=self.run_config.batch_size,
            num_train_epochs=self.run_config.num_epochs,
            weight_decay=self.run_config.weight_decay,
            fp16=self.run_config.fp16,
            push_to_hub=False,
            learning_rate=self.run_config.lr,
            report_to=["none"],
            warmup_ratio=self.run_config.warmup_ratio,
            load_best_model_at_end=True,
            label_names=["protein_input_ids"],  # hack fix for 'eval_loss' not found error
            deepspeed=self.run_config.deepspeed_config,
            local_rank=self.run_config.local_rank,
            gradient_checkpointing=True,
        )

    def build_trainer(self):
        return CLIPPretrainTrainer(
            model=self.task_model,
            args=self.train_args,
            data_collator=DataCollatorForETPretrain(self.protein_tokenizer,
                                                                 self.text_tokenizer,
                                                                 mlm_probability=getattr(self.run_config,
                                                                                         "mlm_probability", 0.0)),
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
            protein_model_fixed=self.run_config.protein_model_fixed,
            text_model_fixed=self.run_config.text_model_fixed,
            lr_ratio=self.run_config.lr_ratio,
        )