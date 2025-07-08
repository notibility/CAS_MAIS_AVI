from openai import OpenAI
import os
import glob
import shutil
import json

# DeepSeek API配置
API_KEY = "your_deepseek_api_key"
BASE_URL = "https://api.deepseek.com"

def call_deepseek_api(prompt):
    """调用DeepSeek API"""
    try:
        print(f"发送的提示词长度: {len(prompt)}")
        print(f"提示词前100字符: {prompt[:100]}...")
        
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=8192,
            temperature=0.7,
            stream=False
        )
        
        print(f"API响应状态: 成功")
        print(f"响应对象类型: {type(response)}")
        print(f"响应内容长度: {len(response.choices[0].message.content)}")
        print(f"响应内容前100字符: {response.choices[0].message.content[:100]}...")
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"API调用失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
        return ""

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def process_file(file_path, output_base_dir):
    # 读取文件内容
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 从文件名中提取问题类型编号
    file_name = os.path.basename(file_path)
    question_type = file_name.split('_')[-1]  # 获取文件名最后一部分，如 q1, q2 等
    
    # 构建提示词
    prompt = f"""【Role Positioning】​​ You are a professional personality psychology researcher with expertise in natural language processing technology.
​​【Task】​​ Evaluate personality traits and provide behaviorally anchored summaries for the given interview text. The current text is a response to the {question_type[1]} category of questions.
Use labels q1 to q6 to annotate your evaluations, which will serve as enhanced text inputs for subsequent model training. For example, 5a03d20a7ecfc50001be0a7a_q1 represents the input for the first question. Therefore, you only need to analyze the response based on the scoring criteria for the {question_type[1]} question. Keep the analysis within approximately 200 words.

**Prompt:**

1. **Openness to Experience (O)** – Aesthetic Appreciation, Inquisitiveness, Creativity, Unconventionality  
   - **Score 1 (Low):** Avoids unfamiliar ideas, uninterested in arts or abstract topics.  
   - **Score 3 (Medium):** Occasionally explores new ideas but prefers practical solutions.  
   - **Score 5 (High):** Seeks out novel experiences, enjoys creative thinking, challenges conventional views.

2. **Emotionality (E)** – Fearfulness, Anxiety, Dependence, Sentimentality  
   - **Score 1 (Low):** Remains detached in emotionally intense situations, avoids emotional expression.  
   - **Score 3 (Medium):** Occasionally expresses emotion and seeks support when stressed.  
   - **Score 5 (High):** Easily overwhelmed by stress, frequently seeks emotional reassurance and support.  

3. **Honesty-Humility (H)** – Sincerity, Fairness, Greed Avoidance, Modesty  
   - **Score 1 (Low):** Exaggerates accomplishments, manipulates others for personal gain, takes credit for others' work.  
   - **Score 3 (Medium):** Generally fair but may exaggerate minor achievements or avoid blame.  
   - **Score 5 (High):** Always honest, shares credit, refuses unethical shortcuts even under pressure.  

4. **Extraversion (X)** – Social Self-Esteem, Social Boldness, Sociability, Liveliness  
   - **Score 1 (Low):** Avoids social interaction, speaks only when addressed, uncomfortable in groups.  
   - **Score 3 (Medium):** Participates in discussions when prompted, occasionally initiates conversation.  
   - **Score 5 (High):** Actively engages with others, enjoys attention, frequently initiates group interactions.  

5. **Agreeableness (A)** – Forgiveness, Gentleness, Flexibility, Patience  
   - **Score 1 (Low):** Easily irritated, responds defensively to criticism, often holds grudges.  
   - **Score 3 (Medium):** Accepts feedback with some hesitation, sometimes argues but moves on quickly.  
   - **Score 5 (High):** Handles conflict calmly, quickly forgives mistakes, consistently cooperative.  

6. **Conscientiousness (C)** – Organization, Diligence, Perfectionism, Prudence  
   - **Score 1 (Low):** Frequently misses deadlines, forgets details, and avoids long-term planning.  
   - **Score 3 (Medium):** Usually meets deadlines, but may overlook minor details or need reminders.  
   - **Score 5 (High):** Always punctual, highly organized, double-checks work, plans proactively.  

Output requirements:
   - Don't add any labels or marks.
   - The output text uses the same language as the original text.
   - No need to provide a score. Just give suggestions and an analysis of the person's character.
   - For each output answer of a txt file, do not provide an analysis for all six personalities. Just give the analysis corresponding to the type of question that the text belongs to.
   - Don't include unnecessary prefixes, such as the prefix for question types, and simply output the pure text of the analysis.

The original text is below：
{content}

Please provide the evaluation text directly, without including any other content："""

    # 调用DeepSeek API
    content = call_deepseek_api(prompt)
    print(f"生成的文本长度: {len(content)}")
    print(f"生成的文本内容: {content[:100]}...")  # 打印前100个字符用于调试

    # 构建输出文件路径
    rel_path = os.path.relpath(file_path, input_base_dir)
    output_dir = os.path.join(output_base_dir, os.path.dirname(rel_path))
    ensure_dir(output_dir)
    
    # 保存增强后的文本（使用与原文本相同的文件名）
    output_file = os.path.join(output_dir, os.path.basename(file_path))
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

def process_directory(input_dir, output_dir):
    """处理目录下的所有txt文件，保持目录结构"""
    global input_base_dir
    input_base_dir = input_dir
    
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 获取所有txt文件
    txt_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.txt'):
                txt_files.append(os.path.join(root, file))
    
    # 处理每个文件
    total_files = len(txt_files)
    for i, file_path in enumerate(txt_files, 1):
        print(f"处理文件 [{i}/{total_files}]: {file_path}")
        process_file(file_path, output_dir)
        print(f"完成处理: {file_path}")

if __name__ == "__main__":
    # 指定输入和输出目录路径
    input_directory = "temp-trascripts"  # 请替换为实际的输入目录路径
    output_directory = "deepseek-r1"  # 请替换为实际的输出目录路径
    process_directory(input_directory, output_directory)