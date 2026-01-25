- 针对local_data的Album_A和univer-light下面（包括子文件夹的）所有图片 进行人脸和人体检测
    - 推荐使用mediapipe进行检测
    - 每张图仅仅保存最大的人脸和人体就可以
    - 结果保存在local_data/{folder_name}_detect_result.jsonl就可以
- 开发的时候可以测试一两张图片渲染到 local_data/visualization 然后查看下坐标是否正确
- 测试成功之后再进行批量运行
- 批量运行需要有tqdm的进度条
- 我希望针对detect的代码主要被限制在src/detect文件夹下

# 新增修改

我发现最大人脸有时候和body不相交，额外检查一下哪些人脸有60%以上的面积在最大body中，额外增加一个backup_face的字段，保存在body中最大的那个人脸。