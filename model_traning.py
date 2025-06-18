from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
import os
import json
import torch

def train_story_model(dataset_path, output_dir="./story_gpt2", model_path="/home2/zzl/nlp_product/gpt2"):
    # 加载safetensors模型
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(
        model_path,
        use_safetensors=True,
        trust_remote_code=False
    )
    
    # 添加特殊标记和pad_token
    special_tokens = {
        "additional_special_tokens": [
            "<|KEYWORDS|>", "<|EMOTION|>", "<|STORY|>", "[PAD]"  # 新增[PAD]标记
        ]
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # 手动设置pad_token（关键修改）
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    
    # 自定义数据集类（包含关键词和情感标签）
    class ConditionalStoryDataset(torch.utils.data.Dataset):
        def __init__(self, file_path, tokenizer, max_length=512):
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.examples = []
            
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    keywords = ", ".join(data["keywords"])
                    emotion = data["emotion"]
                    story = data["text"]
                    input_text = f"<|KEYWORDS|> {keywords} <|EMOTION|> {emotion} <|STORY|> {story}"
                    self.examples.append(input_text)
        
        def __len__(self):
            return len(self.examples)
        
        def __getitem__(self, idx):
            text = self.examples[idx]
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding.input_ids[0],
                "attention_mask": encoding.attention_mask[0]
            }
    
    # 初始化数据集
    train_dataset = ConditionalStoryDataset(dataset_path, tokenizer)
    
    # 使用DataCollatorForLanguageModeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # 训练参数配置
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        save_steps=5000,
        save_total_limit=2,
        prediction_loss_only=True,
        fp16=True,
        report_to="none"
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )
    
    # 开始训练
    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    print(f"模型已保存到 {output_dir}")

if __name__ == "__main__":
    train_story_model("/home2/zzl/nlp_product/stories_dataset.jsonl")