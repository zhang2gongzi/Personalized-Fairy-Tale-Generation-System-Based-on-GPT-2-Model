import os
import pandas as pd
import json
import re


def prepare_story_dataset(data_folder, output_path="stories_dataset.jsonl"):
    # 关键词和情感标签映射
    emotion_keywords = {
        "冒险": ["冒险", "危险", "挑战", "探索"],
        "温馨": ["家庭", "朋友", "爱", "温暖"],
        "悬疑": ["神秘", "秘密", "谜题", "侦探"],
        "科幻": ["机器人", "宇宙", "未来", "科技"]
    }

    all_data = []
    # 遍历文件夹及其子文件夹
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                df = pd.read_csv(file_path)
                all_data.append(df)

    # 合并所有数据
    combined_df = pd.concat(all_data, ignore_index=True)

    stories = []
    for _, row in combined_df.iterrows():
        text = row.get('text')
        if pd.isna(text):
            continue
        text = str(text)
        # 简单分割故事段落
        paragraphs = re.split(r'\n{2,}', text)

        for para in paragraphs:
            if len(para) > 100:  # 只保留足够长的段落
                # 提取关键词和情感标签
                keywords = []
                emotion = "日常"

                for emo, keys in emotion_keywords.items():
                    if any(key in para for key in keys):
                        emotion = emo
                        keywords.extend(keys)
                        break

                # 构建样本
                stories.append({
                    "text": para,
                    "keywords": list(set(keywords)),  # 去重
                    "emotion": emotion
                })

    # 保存为 JSONL 格式
    with open(output_path, "w") as f:
        for story in stories:
            f.write(json.dumps(story) + "\n")

    print(f"数据集已保存到 {output_path}，共 {len(stories)} 个样本")


if __name__ == "__main__":
    data_folder = "/home2/zzl/nlp_product/section-stories"
    prepare_story_dataset(data_folder)