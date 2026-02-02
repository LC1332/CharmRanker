# 编写classify+

在src/ImageNormalizer 下面我已经写好了一个图片归一化的程序

但是我在实际搭建颜值比较的时候遇到一个问题

我图片实际上要先分类男性和女性 然后才能进行颜值比较

这里性别分类我不希望使用MLLM来进行了，
“TensorFlow 在这个环境中似乎有兼容性问题。不要用deepface”
先试试insightface 再看看别的方案

我希望建立一个函数

detect_with_gender( image, require_gender = 'any' )

当require_gender='male' 的时候 只返回男性的face和body的bounding box信息
当等于 female的时候 只返回女性的face和body的bounding box信息
等于any的时候 同时返回两个性别的信息（另外有额外的gender字段表示性别）
(detect可以参考normalizer里面的实现)

我希望你新实现的代码尽量只限制在src/gender_and_normalize文件夹中

# 测试

在local_data/test下建立一个文件夹 gender_and_normalize
结果存在这里面

并且尝试在local_data/claw_by_minimax/celeb_female 中
随机找5张图片测试 require_gender='female'

在local_data/claw_by_minimax/celeb_male 中
随机找5张图片测试 require_gender='male'


# 开始crop

我注意到insightface是以对脸进行性别分类为主
那我们就检测脸就可以了

这里我希望把local_data/claw_by_minimax/celeb_female下每一张照片
中间最大女性人脸找出来，然后 使用ImageNormalizer的.normalize( face_bbox ) 输入
来把图crop出来 先帮我测试5张 保存在local_data/test下

这里我希望把local_data/claw_by_minimax/celeb_male下每一张照片
中间最大男性人脸找出来，然后 使用ImageNormalizer的.normalize( face_bbox ) 输入
来把图crop出来 先帮我测试5张 保存在local_data/test下

# 重构和再测试

既然你自己实现了逻辑

我需要把config.yaml 提取出来
然后有一个类 可以实现 .normalize( image_path )的使用
并且要保留是否渲染body box的开关


# 批量运行

我已经把config调整成渲染红色body框
我现在要进行批量运行了

这里我希望把local_data/claw_by_minimax/celeb_female下每一张照片
中间最大女性人脸找出来，然后 使用ImageNormalizer的.normalize( face_bbox ) 输入
来把图crop出来 保存到local_data/univer_female_celeb
然后按照local_data/univer_female.jsonl的格式写一个univer_female_celeb.jsonl

把local_data/claw_by_minimax/celeb_male下每一张照片
中间最大男性人脸找出来，然后 使用ImageNormalizer的.normalize( face_bbox ) 输入
来把图crop出来 保存到local_data/univer_male_celeb
然后按照local_data/univer_male.jsonl的格式写一个univer_male_celeb.jsonl

