我主要想对src/classify_batch的结果进行 统计 和 查看

注意到结果都在local_data/crop_classify.jsonl 中

注意输出的category有 "gender": "female", "if_asian": "yes", "if_ambiguous": "no", "if_correct_face": "yes", "if_frontal": "yes", "false_alarm": "no" 这些

我希望渲染一个页面

第一个tab上会展示各种category的边缘分布

第二个tab 我可以对每个category进行选择（筛选或者不选择） 比如支持选择gender = femal if_asian = yes的所有图

以及一个按钮 来刷新随机展示5张crop后的图片

限制你生成的代码都在src/classify_vis中

---

## 使用方法

```bash
# 在项目根目录运行
python3 src/classify_vis/app.py
```

然后在浏览器中打开 http://127.0.0.1:5002

### 功能说明

1. **分布统计 Tab**: 展示各个category的边缘分布饼图，包括gender、if_asian、if_ambiguous、if_correct_face、if_frontal、false_alarm

2. **筛选查看 Tab**: 
   - 可以对每个category进行筛选（选择特定值或"全部"）
   - 点击"刷新随机样本"按钮随机展示5张符合筛选条件的crop图片
   - 点击"重置筛选"按钮清空所有筛选条件