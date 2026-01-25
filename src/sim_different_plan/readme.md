我有一个多item比较来确定每个item latent score的BTL问题

假设每个item的latent在1到100之间均匀分布

然后给定三个物体，我可以调用一次API 获得三个物体的程度比较（最大和最小）

这个建模已经在 src/sim_triple_n_vs_error/simulate_triplet.py 中进行了实现

现在的问题是这样的，我有N = 5000 个样本

但是我的budget是 30000次api调用。我可以进行30000次的api比较

我希望所有样本的rank error尽可能的小

## 方案1

平均比较 每个样本调用一样的次数

## 方案2

将样本划分为 N_anchor = 200大小的 Anchor set 和 4800 大小的 Pool set

然后每次都以(Anchor, Anchor, pool) 的组合 进行比较
这个方法旨在 得到一个稠密的anchor 集合 这个anchor集合的序是想对稳定的

## 方案3

测试一下方案2中 anchor set 的大小为N_anchor=500的情况

## 方案4

测试 N_anchor = 1000 的情况

## 方案5 动态调整

先进行10000次api调用 计算一下分数

然后将所有item的分数根据rank 归一化到 0 - 100

之后每次再选取 500个triplet， 并且triplet的三个item的最大差保持在 threshold = 20以内

循环直到所有的api budget被花完

## 方案6

和方案5相同 threshold = 40


比较的时候增加tqdm展示 
保持你的新增代码尽量只在src/sim_different_plan中