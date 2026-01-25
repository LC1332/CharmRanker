# Batch API的测试

针对 src/classify/batch_classify.py
我发现一条一条进行gemini询问还挺慢的
我想试试公司的api能不能使用gemini batch

我希望你阅读gemini-batch-doc.md

结合crop_log_with_face_and_body.jsonl和crop_classify.jsonl

选取倒数10条还没有完成classify的数据

尝试调用gemini的batch api（仍然使用gemini-3-flash-preview模型）

如果结果正确的话，增量写入到crop_classify.jsonl中

# 批量运行

我注意到只有公司api key的时候只能使用并发加速

那你使用并发来完成crop_classify.jsonl中未完成的部分吧。