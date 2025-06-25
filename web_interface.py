import gradio as gr
from story_generator import StoryGenerator

# 初始化故事生成器
generator = StoryGenerator()

def generate_story(keywords, emotion, story_length):
    # 将滑块值转换为最大长度
    max_length = int(story_length * 100)
    # 处理关键词输入
    keywords_list = [k.strip() for k in keywords.split(",") if k.strip()]
    
    if not keywords_list:
        return "请至少输入一个关键词"
    
    story = generator.generate_story(keywords_list, emotion, max_length)
    return story

# 创建Gradio界面
iface = gr.Interface(
    fn=generate_story,
    inputs=[
        gr.Textbox(label="关键词（用逗号分隔，如：森林,魔法,公主）"),
        gr.Radio(["adventure", "warmth", "suspense", "magic","happy"], label="情感倾向"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="故事长度（百字）")
    ],
    outputs=gr.Textbox(label="生成的故事"),
    title="个性化故事生成系统",
    description="输入关键词和情感倾向，AI将为你创作一个独特的故事！"
)

# 启动界面
iface.launch()