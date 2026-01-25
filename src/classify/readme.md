# 函数实现

- 我希望变现一个函数 classify_gender_and_asian
    - 输入是一个图片的路径
    - 输出是一个完整的json 包含
        - gender 是男性还是女性
        - if_asian 是否是亚洲人
        - 更多其他字段的信息

我希望使用一个gemini的prompt来进行这样的分类

请使用local_data/visualization/cropped_sample下的两张照片来进行测试

gemini使用公司做了统一的api转发
    - key见 整个项目的.env的LUMOS_API
    - api的使用说明见api_help.md
    - 我之前有一个跑通的例子见test_detection.py

prompt 我希望使用这样一个prompt 请参考我的中文原文来写一个英文的prompt

```
请根据我输入的图片，对图片中红色框框出的人，进行性别等attribute的判断

并以json格式输出以下字段

- analysis: 对于图片中的红框是否确定了主体人物，是否是亚洲人，是什么性别等进行初步的分析
- gender: 男性输出male 女性输出female 误检或者无法确认的情况输出 unpredictable
- if_ambiguous: 画面中的红色框，是否较为准确的确定了唯一一个人 如果是 输出 yes 如果有一个人被红色框大部份框中，其他人只是出现了部分，仍然输出yes 如果bounding box明显框中了两个人，则输出 no
- if_correct_face: 对于画面中的绿色人脸框，是否是红色框指定的那个人，如果是输出yes 如果不是 输出no
- if_frontal: 画面中的人是否正脸或者侧脸面向镜头，如果能看到五官则输出 yes 如果不是 则输出 no
- false_alarm: 画面中是否一个人都没有，即红色框没有框中人是个误检，如果是 则输出yes 如果画面中有人则输出no
```

你可以对我的prompt翻译成英文 并且进行合适的polish 

测试cropped_sample下的两张照片, 看看结果是否能正确的被json解析

确保你新增的代码都只在src/classify中

# 批量处理

对local_data/crop_log_with_face_and_body.jsonl 中涉及到的所有crop后的图片
（他们实际上被存储在local_data/crop_output中）
进行处理 对于读取图片你只需要考虑output_filename 字段
但是输出jsonl的时候请帮我保留所有的其他字段

输出到local_data/crop_classify.jsonl中，把新的json解析结果都加入到里面
增加tqdm显示

## 关于重试
每张图允许重试1次
都失败的图信息存储到local_data/fail_to_classify.jsonl中

## 关于重开
每次重开的时候会读取local_data/crop_classify.jsonl 对于已经顺利保存的图片（分析输入文件名） 直接跳过


# 把属性合适的分离

检查local_data/crop_classify.jsonl 

我只需要保留 if_asian = yes
if_ambiguous = no
if_frontal = yes 的数据

我想把 local_data/crop_body_only/{output_filename}

根据gender分别保存到 local_data/univer_male 和 local_data/univer_female 中

并且把对应的jsonl行 建立到 local_data/univer_{gender}.jsonl 中

