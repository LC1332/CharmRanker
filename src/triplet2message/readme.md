
# 颜值的triplet2message

我需要使用gemini等MLLM来比较三张照片中人物的颜值

我希望有一个triplet2message 函数生成message

然后再有一个通用的抽象接口 可以帮我把 message通过inference MLLM变为json输出

## 参考

这里重点可以参考 step4_annotate_triplets.py里面的 
build_annotation_message 和 parse_annotation_response 函数

## 关于openai和gemini的转发

openai和gemini的转发可以参考 src/diandian/api_help.md 
并且在src/classify/classify中已经给出了一个实现
不要在代码中暴露baseurl或者key

## 关于测试

先帮我测试一组triplet就可以
图片我已经放在local_data/test/triplet中间
跑通gemini和openai
你测试结果如遇要存 也存在这个文件夹

## prompt

请使用下列的prompt
    
prompt = f"""Please analyze these THREE images and identify which person is the MOST attractive and which person is the LEAST attractive.

Consider both the person's appearance/styling and how well the photo presents them when making your judgment.

The person to be evaluated in each image has been highlighted with a red bounding box.

Please examine all three images carefully:
- Image A (first image)
- Image B (second image)
- Image C (third image)

**IMPORTANT: You MUST respond with ONLY a valid JSON object, no other text.**

Output Format (copy this structure exactly):
{
  "analysis": "Give a comprehensive analysis that you can detailedly comparing all three images",
  "most_attractive": "A" or "B" or "C" or "unpredictable",
  "least_attractive": "A" or "B" or "C" or "unpredictable"
}

- Set "most_attractive" to the letter of the image with the MOST attractive person.
- Set "least_attractive" to the letter of the image with the LEAST attractive person.
- Use "unpredictable" only if you genuinely cannot make a confident judgment.

Now analyze the three images and respond with ONLY the JSON object:
"""

- 尽量把实现的代码控制在src/triplet2message文件夹中

# 修改prompt

我发现现在的prompt有点太依赖于照片的环境和衣着了

保留之前的prompt版本为 prompt_v0.md 然后修改prompt
在prompt前面加入“你是一个星探，在为公司选取偶像候选人或者年会主持人”（记得翻译成英文）
在prompt中间去掉
Consider both the person's appearance/styling and how well the photo presents them when making your judgment.
改为
更强调person本身的颜值和身材的attrative

