我有一个多item比较来确定每个item latent score的BTL问题

假设每个item的latent在1到100之间均匀分布

给定两个item 
然后我有个比较分类器，这个分类器会比较两个物体

如果我比较两个物体

分对的概率 相对于他们的latent score的差呈现一个logistic 分布

我知道 在latent差=10的时候 分对的概率为73% 
（latent差=0的时候 判任意一边大的概率为50%）
(这个设定和src/sim_trueskill_n_vs_error中的相同)


但是现在 我使用一种"triplet比较"的方法

我们每次会比较3个物体，要求annotator标出其中的最大 或/和 最小的样本
也就是一次会比较3条边
请帮我根据这个标注方式做合适的概率建模
（给出一次triplet标注的3条边的概率方案）

我现在假设我有N = 200个样本

之前 K = 4、8、16、32、...64 次分别会调用 KN/2 次api
我现在希望仍然研究 平均调用 K 次api （ 也就是K' = 2,4,... 32 ）
实际会进行K'N次triplet比较 产生 3K'N 条边

的情况下

K' vs rank error的关系 

图 放在 local_data/visualization/trueskill_simu中
jpg格式保存

限制你的代码只在sim_triple_n_vs_error文件夹中