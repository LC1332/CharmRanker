帮我建立的完整的程序
进行动态的图片的颜值比较 使用glm-4.6v模型

src/EloManager中实现了EloManager类
- 这个类可以动态管理node的分数
- 以及propose需要测试的triplet案例

src/triplet2message中实现了glm的MLLM inference
- 已经进行了glm 4.6v模型的颜值inference实现
- 但是注意到我还没有做并行的实现

# 数据和输出
这次实验数据的输入数据图片是local_data/univer_female 帮我随便sample个300张进行实验
所有的log都帮我输出到local_data/glm_female 中

# 并行机制

在config中定义 MAX_WORKER = 15（设置在config） 表示最大允许的worker数量
K_ONE_TIME = 5（设置在config） 表示一次会调用elo manager 的 propose_triplet 方法的K

当空余的worker数大于K_ONE_TIME 的时候，会要求manager propose一次triplet 并且发出

第一次不会等待 之后每次批量发出会等待5秒（等待时间也设置到config），并且检查空余worker是不是大于K_ONE_TIME个
如果等待超过3分钟 没有worker完成，则每3分钟print一次报警

如果worker得到了glm的response，则会调用elomanager的方法 更新节点的latent score

# 关于config

建立一个config.yaml 记录这次试验所有的elomanager的log
定义 save_path = "local_data/glm_female"
定义 sample_n = 300 如果sample_n 不定义 则使用文件夹中的所有图片

# 关于budget

这次帮我调用个 1800 次 api就可以了。

# response log
我需要记录所有glm的response log
即参与了triplet的是哪几个图片 模型的response是啥 如果json不能正常解析 则把raw_response存储下来

- 限制你的代码尽量都在 src/dynamic_glm中

# 关于断点续传

我程序随时有可能关闭和重启

确保config稳定的时候重启程序可以继续进行
