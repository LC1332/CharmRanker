# 总体需求

在 src/sim_dynamic 中 我已经验证了 如果我有N个节点要比较的情况下

使用triplet，先保证每个节点参与了2次比较，然后再每次动态比较K个

可以获得相对更好的准确率

所以我希望建立一个EloManager 来管理node和edge的信息 以及propose新的triplet用于标注

在这次开发中 我希望主要是用模拟比较的方式 来验证EloManager的正确性

# EloManager的接口设计

## save( path )

可以将 节点和比较log的信息 保存到 path 中

这里每个节点会保存其对应的 json形式的info 以及增加一个额外的字段 score 节点会用 name字段来进行唯一索引

比较log。 这里我们也会在path里面记录 triplet的比较结果

## load( path )

这里因为我很有可能在之后MLLM比较的时候会随时断开和重新比较

这里我们要支持从path 去load 结果

## add_nodes( node_infos )

输入一个list of json 每个元素保存着node的信息。 如果没有name元素，会生成一个hash来作为node的name。并且初始化对应的score

## propose_triplet( K = 10 )

这里会根据 (Triplet Propose 规则) 中的规则，来给出K个propose的triplet

返回 K 大小的一个list 每个元素包含了要比较的3个node的信息

## submit_triplet_result( results, triplet_nodes )

这里results会输入一个list， 里面每个元素包含 largest 和 smallest 字段 （对应ABC）
说明这次标注模型/模拟 觉得triplet里面是第1、2、3个元素被标记为了最大和最小

## 关于config

请把很多超参数的设置放在一个config.yaml中



# Triplet Propose 规则

## 有重复不抽原则

对于一个(A,B,C) triplet 如果任何一个顺序(比如(A,C,B))已经被标记过，则不再会被抽样到
（权重*0）

## 单节点温饱原则

如果参与triplet的节点 有 总参与次数 <  self.minimal_compare_times (config中设置为2)
则会优先让这些节点参与比较。 
(优先非温饱节点和非温饱节点比较，这样可以快速让更多的节点去满足比较次数。也允许已经温饱的节点参与比较)

## 邻近抽样原则

对于已经满足self.minimal_compare_times次比较的三个节点

在latent根据rank归一化到0-100以内之后

我们保证抽样出来的(A,B,C) 中的 |最大分位数 - 最小分位数| < self.rank_threshold(config中设置为25)

## 高分多抽原则

如果已经满足了温饱原则，要根据邻近抽样原则的基础上，我们会先构造出2倍的候选的triplet

然后计算(A,B,C)的调和平均数。 然后平均数接近0的时候权重是1， 接近100的时候权重是self.beauty_weight(config中设置为3) 进行加权抽样最后返回额定的triplet个

保证 propose_triplet 总是会返回K个


# 关于模拟

## 模拟的latent_score

模拟的时候 每个node除了name，额外会有一个latent_score的信息

## 模拟triplet比较

我之后会正式介入MLLM的比较
但是在此之前，先试用src/sim_dynamic 的模拟比较函数
会根据latent_score 根据概率给比较结果

我的输入有4000个节点，每次就标记K = 10个吧。

然后进行3000个循环，每30个循环查看一下平均rank error的变小情况


# 更多测试

写一个新的测试

帮我模拟一下初始有4000个节点

然后运行3000个循环，每30个循环查看一下平均rank error的变小情况

然后每30个循环加入1个新节点的情况

数据仍然存储到local_data/test_elo_manager (可以覆盖之前的jsonl)