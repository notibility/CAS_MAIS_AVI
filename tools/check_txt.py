import os

def check_empty_txt_files(directory):
    empty_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if not content:
                        empty_files.append(file_path)
    if empty_files:
        print("以下txt文件为空：")
        for f in empty_files:
            print(f)
    else:
        print("没有空白的txt文件。")

# 用法示例
check_empty_txt_files("/root/autodl-tmp/yanglongjiang/project/AVI/7-1wenbenzengqiang-deepseek-r1")