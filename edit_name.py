import re
import os

path = 'C:\\3.12 qCMOS\\full_200ms_4530shots'


# 创建一个正则表达式来匹配文件名
pattern = re.compile(r"(Background001_)(\d+)( \(2\).tif)")

# 获取目录下的所有文件
filenames = os.listdir(path)



for filename in filenames:
    # 匹配文件名
    match = pattern.match(filename)

    if match:
        # 获取匹配的数字
        number = int(match.group(2))
        # 计算新的数字
        new_number = number + 4530
        # 生成新的文件名
        new_filename = f"Background001_{new_number:05}.tif"
        # 重命名文件
        os.rename(os.path.join(path, filename), os.path.join(path, new_filename))