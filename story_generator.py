from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import re
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter

# 确保nltk数据已下载
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
        self.end_token = "<|END|>"  # 新增结束标记
        
        # 故事结构模板（可选）
        self.story_templates = {
            "adventure": "Once upon a time, {protagonist} set out on a journey to {goal}. Little did they know that this journey would lead them to {conflict}. Along the way, they met {ally} who helped them face {obstacle}. In the end, they discovered that {resolution}.",
            "warmth": "In a small village, {protagonist} lived a simple life with {family/friends}. One day, {inciting_event} brought {change} into their world. Through this experience, they learned the importance of {theme} and how {relationship} could overcome any challenge.",
            "suspense": "When {protagonist} stumbled upon {mysterious_object/clue}, they unwittingly became entangled in {mystery}. As they dug deeper, they uncovered a web of lies involving {villain}. With time running out, {protagonist} must {action} before {consequence}."
        }
    
    def generate_story(self, keywords, emotion, max_length=500, temperature=0.75, num_return_sequences=1, 
                       no_repeat_ngram_size=3, repetition_penalty=1.5, top_k=40, top_p=0.85, 
                       use_template=False, structure_penalty=0.9):
        """生成故事并优化重复问题"""
        # 构建条件输入
        base_prompt = f"{self.keywords_token} {' '.join(keywords)} {self.emotion_token} {emotion} {self.story_token}"
        
        # 使用故事模板（可选）
        if use_template and emotion in self.story_templates:
            # 简单填充模板
            template = self.story_templates[emotion]
            filled_template = template.format(
                protagonist="a brave soul",
                goal="discover a hidden treasure",
                conflict="a dangerous challenge",
                ally="a wise old mentor",
                obstacle="a series of tests",
                resolution="true strength comes from within"
            )
            prompt = f"{base_prompt} {filled_template}"
        else:
            # 标准提示
            genre_hint = {
                "adventure": "A thrilling adventure story where",
                "warmth": "A heartwarming tale about",
                "suspense": "A mysterious story filled with suspense and intrigue"
            }
            hint = genre_hint.get(emotion.lower(), "A story about")
            prompt = f"{base_prompt} {hint} "
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # 计算需要生成的新token数量
        new_tokens = max_length - input_ids.shape[1]
        if new_tokens <= 0:
            new_tokens = 150  # 默认生成150个新token
            max_length = input_ids.shape[1] + new_tokens
        
        # 修复：检查特殊标记是否在词汇表中
        end_token_exists = self.end_token in self.tokenizer.get_vocab()
        eos_token_id = self.tokenizer.encode(self.end_token, add_special_tokens=False)[0] if end_token_exists else None
        
        # 生成故事
        output = self.model.generate(
            input_ids,
            max_length=max_length,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            do_sample=True,
            length_penalty=structure_penalty  # 控制故事结构
        )
        
        # 解码并提取故事部分
        generated_stories = []
        for out in output:
            generated_text = self.tokenizer.decode(out, skip_special_tokens=False)
            story_start = generated_text.find(self.story_token) + len(self.story_token)
            story = generated_text[story_start:].strip()
            
            # 截断到结束标记（如果存在）
            if self.end_token in story:
                story = story[:story.index(self.end_token)].strip()
            
            # 优化故事质量
            story = self._optimize_story(story, keywords)
            
            generated_stories.append(story)
        
        return generated_stories[0] if num_return_sequences == 1 else generated_stories
    
    # 以下方法保持不变...
    def _optimize_story(self, text, keywords):
        """优化故事文本的连贯性和可读性"""
        # 1. 修复基本格式问题
        text = self._fix_formatting(text)
        
        # 2. 检测并截断重复内容
        text = self._truncate_repetitive(text)
        
        # 3. 确保关键词出现（如果可能）
        text = self._ensure_keywords(text, keywords)
        
        # 4. 增强故事结构
        text = self._enhance_structure(text)
        
        return text
    
    def _fix_formatting(self, text):
        """修复基本格式问题"""
        # 规范化引号
        text = re.sub(r'\'\'|``', '"', text)
        
        # 修复多重空格
        text = re.sub(r'\s+', ' ', text)
        
        # 确保段落分隔合理
        paragraphs = re.split(r'\n\s*\n', text)
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para:
                cleaned_paragraphs.append(para)
        
        return '\n\n'.join(cleaned_paragraphs)
    
    def _truncate_repetitive(self, text):
        """检测并截断重复段落"""
        if not text:
            return text
            
        # 使用nltk更准确地分割句子
        try:
            sentences = sent_tokenize(text)
        except:
            # 回退到简单分割
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
        if len(sentences) < 4:
            return text  # 短文本不处理
            
        # 检测重复句子
        filtered_sentences = []
        prev_sent = ""
        repeat_count = 0
        
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
                
            # 检查与前一句的相似度
            similarity = self._sentence_similarity(sent, prev_sent)
            if similarity > 0.7:
                repeat_count += 1
                if repeat_count >= 2:  # 允许少量重复
                    continue
            else:
                repeat_count = 0
                
            filtered_sentences.append(sent)
            prev_sent = sent
            
        return ' '.join(filtered_sentences)
    
    def _ensure_keywords(self, text, keywords, min_keywords=1):
        """确保故事中包含一定数量的关键词"""
        if not keywords:
            return text
            
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in keywords if keyword.lower() in text_lower)
        
        # 如果关键词不足，尝试添加
        if keyword_count < min_keywords:
            missing_keywords = [kw for kw in keywords if kw.lower() not in text_lower]
            if missing_keywords:
                # 在故事开头添加缺失的关键词
                first_sentence_end = text.find('.')
                if first_sentence_end > 0:
                    insert_pos = min(first_sentence_end, 100)  # 在开头100个字符内插入
                    text = f"{text[:insert_pos]} {', '.join(missing_keywords)} {text[insert_pos:]}"
                    
        return text
    
    def _enhance_structure(self, text):
        """增强故事结构，确保有开头、发展和结尾"""
        # 简单结构检测
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 如果段落太少，尝试添加分隔
        if len(paragraphs) < 3 and len(text) > 300:
            # 尝试在合理位置分割
            mid_point = len(text) // 2
            # 找到最近的句号
            period_pos = text.rfind('.', mid_point - 50, mid_point + 50)
            if period_pos > 0:
                paragraphs = [text[:period_pos+1], text[period_pos+1:]]
        
        # 检查是否有明显的结尾
        if len(paragraphs) > 0 and not re.search(r'[.!?]$', paragraphs[-1]):
            # 尝试添加结尾
            last_sentence = paragraphs[-1].strip()
            if not re.search(r'[.!?]$', last_sentence):
                paragraphs[-1] = last_sentence + '.'
                
        return '\n\n'.join(paragraphs)
    
    def _sentence_similarity(self, s1, s2):
        """计算两个句子的相似度"""
        if not s1 or not s2:
            return 0
            
        # 简单的词重叠率计算
        s1_words = set(s1.lower().split())
        s2_words = set(s2.lower().split())
        common = s1_words.intersection(s2_words)
        total = s1_words.union(s2_words)
        
        return len(common) / max(1, len(total))