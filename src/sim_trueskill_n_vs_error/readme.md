我有一个多item比较来确定每个item latent score的BTL问题

假设每个item的latent在1到100之间均匀分布

给定两个item 
然后我有个比较分类器，这个分类器会比较两个物体

分对的概率 相对于他们的latent score的差呈现一个logistic 分布

我知道 在latent差=10的时候 分对的概率为73% 
（latent差=0的时候 判任意一边大的概率为50%）

我现在假设我有N = 200个样本

我想研究每个样本在平均参与
K = 4、8、16、32、...64 次比较的之后
通过
src/trueskill/calculate_trueskill_scores.py
计算得到的trueskill得分 的rank
和item本身ground truth rank的差的平均值

（这里你可以shuffle id K次 使用shuffle id1 对 shuffle id 2进行比较 然后去重就可以）

画出 K vs rank_error/N 的图 放在 local_data/visualization/trueskill_simu中
jpg格式保存

确保你的新增代码都在sim_trueskill_n_vs_error文件夹中