我有一个多item比较来确定每个item latent score的BTL问题

假设每个item的latent在1到100之间均匀分布

然后给定三个物体，我可以调用一次API 获得三个物体的程度比较（最大和最小）

这个建模已经在 src/sim_triple_n_vs_error/simulate_triplet.py 中进行了实现

现在的问题是这样的，我有N = 5000 个样本

但是我的budget是 30000次api调用。我可以进行30000次的api比较

我希望所有样本的rank error尽可能的小

我希望实验下面3个方案

## 方案1

平均比较 每个样本调用一样的次数

## 方案2 动态调整

先进行10000次api调用 计算一下分数

然后将所有item的分数根据rank 归一化到 0 - 100

之后每次再选取 500个triplet， 并且triplet的三个item的最大差保持在 threshold = 20以内

循环直到所有的api budget被花完

## 方案3

和方案5相同 threshold = 40


# 加速实现

我之前在src/sim_different_plan中实现了一版

但是我感觉太慢了 主要是因为我true_skill/calculate_trueskill_scores.py 的实现比较慢

注意到trueskill/calculate_true_skill.py中，本来就支持 process_comparisons 增量

所以帮我在dynamic实现里面 改为使用这个增量方案的