from pathlib import Path
import re

current_dir = Path(__file__).resolve().parent

file_names = [f for f in current_dir.iterdir() if f.is_file() and f.suffix == '.log']

input_pattern = r"^test_(\d{14})\.log$"
output_pattern = r"^log_(\d+)\.log$"

input_files = []
output_index = 0

for file_name in file_names:

    match = re.match(input_pattern, file_name.name)
    if match:
        number = int(match.group(1))
        input_files.append((number, file_name))

    match = re.match(output_pattern, file_name.name)
    if match:
        number = match.group(1)
        output_index = int(number) + 1

input_files.sort(key=lambda x: x[0])

if not input_files:
    exit(0)

# 创建输出文件
output_file_path = current_dir / f'log_{output_index}.log'
with open(output_file_path, 'wb') as output_file:
    for _, file in input_files:
        with open(file, 'rb') as input_file:
            # 读取并写入内容
            output_file.write(input_file.read())
            output_file.write(b'\n')  # 可选：在每个文件之间添加换行符

for _, file in input_files:
    file.unlink()  # 删除输入文件

print(f"Input files: {input_files}")
print(f"Output index: {output_index}")