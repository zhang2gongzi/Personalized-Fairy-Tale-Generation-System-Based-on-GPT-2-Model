from datasets import load_metric
import json
import numpy as np
from story_generator import StoryGenerator

def evaluate_model(dataset_path, model_path="./story_gpt2", num_samples=50):
    # 加载评估指标
    perplexity = load_metric("perplexity")
    # bleu = load_metric("bleu")  # 注意：BLEU更适合翻译任务，此处可替换为ROUGE
    
    # 加载模型
    generator = StoryGenerator(model_path)
    
    # 读取数据集样本
    with open(dataset_path, "r") as f:
        lines = f.readlines()[:num_samples]
    
    # 准备评估数据
    references = []
    predictions = []
    texts = []
    
    for line in lines:
        data = json.loads(line)
        keywords = data["keywords"]
        emotion = data["emotion"]
        reference_text = data["text"]
        
        # 生成故事
        generated_story = generator.generate_story(keywords, emotion, max_length=300)
        
        references.append(reference_text)
        predictions.append(generated_story)
        texts.append(generated_story)
    
    # 计算困惑度
    perplexity_results = perplexity.compute(model_id=model_path, input_texts=texts)
    
    # 注意：由于BLEU要求tokenized输入，这里简化处理，实际应使用更适合文本生成的指标
    # bleu_results = bleu.compute(predictions=predictions, references=[[ref] for ref in references], max_order=4)
    
    return {
        "perplexity": perplexity_results["perplexity"],
        # "bleu-4": bleu_results["bleu"],
        "sample_size": num_samples
    }

if __name__ == "__main__":
    results = evaluate_model("stories_dataset.jsonl")
    print("评估结果:")
    for key, value in results.items():
        print(f"{key}: {value}")