import os

TRANSCRIPTS_DIR = 'transcripts'
DEEPSEEK_DIR = '7-1wenbenzengqiang-deepseek-r1'
MERGED_DIR = 'merged_txts_r1'
SPLIT_LINE = '\n\n===\n\n'

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def merge_txt_files(transcripts_dir, deepseek_dir, merged_dir):
    ensure_dir(merged_dir)
    # 获取 transcripts 目录下所有 txt 文件名
    transcripts_files = set(f for f in os.listdir(transcripts_dir) if f.endswith('.txt'))
    deepseek_files = set(f for f in os.listdir(deepseek_dir) if f.endswith('.txt'))
    # 找到两个目录下同名的文件
    common_files = transcripts_files & deepseek_files
    print(f'共找到 {len(common_files)} 个同名文件')
    for filename in common_files:
        transcript_path = os.path.join(transcripts_dir, filename)
        deepseek_path = os.path.join(deepseek_dir, filename)
        merged_path = os.path.join(merged_dir, filename)
        # 读取内容
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_content = f.read()
        with open(deepseek_path, 'r', encoding='utf-8') as f:
            deepseek_content = f.read()
        # 合并内容
        merged_content = transcript_content + SPLIT_LINE + deepseek_content
        # 保存
        with open(merged_path, 'w', encoding='utf-8') as f:
            f.write(merged_content)
        print(f'已合并: {filename}')

if __name__ == '__main__':
    merge_txt_files(TRANSCRIPTS_DIR, DEEPSEEK_DIR, MERGED_DIR) 