src/dynamic_glm或者dynamic_openai给出了动态比较的代码

我现在希望实现一个网页

可以查看动态比较存储在比如local_data/glm_female中的结果

帮我实现一套交互网页的程序 端口5015

config.yaml中可以设置
result_folder = local_data/glm_female
img_folder = local_data/univer_female

# Tab 1

展示 目前top10的节点的照片 包括 mu 和sigma

点击刷新按钮来刷新最新数据

# Tab 2

从response_logs.jsonl中 读取最近进行的5 个比较

显示出哪个图被认为是最attractive 哪个图是least attractive
以及把analysis也显示出来

点击刷新按钮来刷新最新数据


# Tab 3

这里我希望能够调用elo manager的 get_anchors 功能
显示对应的anchors图片。

点击刷新按钮来重新load和调用anchors最新数据
