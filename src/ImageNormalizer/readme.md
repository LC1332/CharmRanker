我要实现一个ImageNormalizer

ImageNormalizer应该是self-contain的 只包含 src/ImageNormalizer 下面自己的代码

ImageNormalizer会根据 src/ImageNormalizer/config.yaml的配置进行启动

ImageNormalizer.normalize 会接受一个image_path 并且输出一个crop_and_blend之后的图（返回为np.ndarray）

normalize相当于会先执行detect (src/detect中的代码)  查找图片中最大的bounding box

## 如果检查到最大的body

则会按照 crop_and_blend的逻辑 进行crop和blend
（不同的是， config只要考虑是否渲染body框 和那些模糊的参数，
 这次我们不需要再渲染face框）
 （并且config中请帮我把if_render_body框设置为false

## 如果没有检查到最大的body 但是检查到了最大的face

这里我们需要通过face来推算最大body的位置

请统计local_data/crop_classify.jsonl 中的数据

只考虑 "if_ambiguous" =  "no", "if_correct_face": "yes", "false_alarm": "no" 的数据

（理论上face框应该是正方形，不行就按照face框的长宽几何平均值作为边长）

计算这些数据中 body box 的 width / face_size, height/ face_size 和 (body_center_y - face_center_y)/face_size 这三个量的平均值 （ x偏移默认为0）

这样我们根据face框的size和中心，就可以推算出body框的位置。

在Album_A_detect_result.jsonl中寻找一张有face框但是没有body框的数据，尝试这个分支的测试

将一个有body框的测试结果和一个没有body框的测试结果 输出到 local_data/visualization/test_normalizer

## 支持body框的输入

ImageNormalizer.normalize 也可以指定一个json格式的body框输入 类似 {"x": 120, "y": 54, "width": 228, "height": 361} 格式

## 支持face框输入 

ImageNormalizer.normalize 也可以指定一个json格式的face框输入 类似 {"x": 120, "y": 54, "width": 228, "height": 361} 格式

## 如果没有框也检测不到
则取图片中心一个最大（边长与图片短边相同）的正方形 作为body框的结果。