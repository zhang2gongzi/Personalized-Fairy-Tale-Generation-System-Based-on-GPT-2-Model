from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class StoryGenerator:
    def __init__(self, model_path="./story_gpt2"):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 设置特殊标记
        self.keywords_token = "<|KEYWORDS|>"
        self.emotion_token = "<|EMOTION|>"
        self.story_token = "<|STORY|>"
    
    def generate_story(self, keywords, emotion, max_length=500, temperature=0.7, num_return_sequences=1):
        # 构建条件输入
        prompt = f"{self.keywords_token} {' '.join(keywords)} {self.emotion_token} {emotion} {self.story_token}"
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 生成故事
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            early_stopping=True
        )
        
        # 解码并提取故事部分
        generated_stories = []
        for out in output:
            generated_text = self.tokenizer.decode(out, skip_special_tokens=False)
            story_start = generated_text.find(self.story_token) + len(self.story_token)
            story = generated_text[story_start:].strip()
            generated_stories.append(story)
        
        return generated_stories[0] if num_return_sequences == 1 else generated_stories