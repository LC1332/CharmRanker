# 核心函数

我希望实现一个 crop_and_blend.py 这个核心函数

输入 - 一个图片的路径， body_bbox (mediapipe格式 规一化后的 x_min y_min width 和 height) 
    - face_bbox (mediapipe格式 实际采用jsonl中的backup_face_bbox字段)

输出 一张正方形的照片（先不要保存下来 而是以内存变量的方式输出） 其中face_bbox

## 数据 

数据在 local_data/Album_A_detect_result.jsonl 和 univer-light_detect_result.jsonl中
我希望只处理body_bbox的短边大于等于100的照片 另外 face框采用backup_face_bbox的框（如有）

## 具体要求

crop and trans:
我需要body_bbox被移动到居中的位置。最长边两边向外padding最长边的15%（如有）
短边方向和长边截取的长度一致
没有图片的部分padding白色。

blending:
我需要用sigma = 5%的长边，将bbox外的景象进行高斯模糊

bbox render
在transfer之后的图上 先用细绿线 渲染backup_face_bbox的框
然后用细红线 渲染bbox框

## 超参数配置

建立一个yaml文件来进行超参数的配置


保证你写的所有文件在src/crop_and_blend文件夹中

## 测试1

随机输出5张结果到 local_data/visualization/crop_and_blend_sample jpg格式
检查至少一张图片确保程序是对的

## 可视化页面

开始编写一个app.py 每次我可以按按钮 随机5张结果来显示在页面上



# 批量化处理

在yaml中增加一个target_folder = "crop_output"中的字段
增加一个 if_render_body = True 的字段
增加一个 if_render_face = True 的字段

编写一个批量处理的程序（带tqdm进度显示） 把所有符合要求（短边大于100）的图片以jpg形式
保存到 local_data/{crop_output} 中， 文件名使用一个hash的方式（以文件名hash）
重新命名成一个10位的英文字符串
    - hash要保证修改 if_render_body等开关的时候文件名不会发生变化
并且记录一个crop_log.jsonl 记录变换后的文件名，face和body的box位置 以及变换前的文件名，和box位置