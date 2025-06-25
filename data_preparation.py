import os
import pandas as pd
import json
import re

def prepare_story_dataset(data_folder, output_path="stories_dataset.jsonl"):
    # 关键词和情感标签映射
    emotion_keywords = {
        "adventure": ["adventure", "brave", "challenge", "exploration"],
        "warmth": ["family", "friends", "love", "warmth"],
        "suspense": ["mystery", "secret", "puzzle", "detective"],
        "magic": ["magic", "witch", "dragon", "unicorn"]
    }

    all_stories = []
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                # 将整个文件内容拼接为一个故事
                story_text = ' '.join([str(row.get('text')) for _, row in df.iterrows() if pd.notna(row.get('text'))])
                # 提取关键词和情感标签
                keywords = []
                emotion = "happy"

                for emo, keys in emotion_keywords.items():
                    if any(key in story_text for key in keys):
                        emotion = emo
                        keywords.extend(keys)
                        break

                all_stories.append({
                    "text": story_text,
                    "keywords": list(set(keywords)),
                    "emotion": emotion
                })

    # 保存为 JSONL 格式
    with open(output_path, "w") as f:
        for story in all_stories:
            f.write(json.dumps(story) + "\n")

    print(f"数据集已保存到 {output_path}，共 {len(all_stories)} 个样本")

if __name__ == "__main__":
    data_folder = "/home2/zzl/nlp_product/section-stories"
    prepare_story_dataset(data_folder)