"""
从有向图边表计算每个节点的 TrueSkill 分数。

通过多次 shuffle 边表来消除输入顺序对结果的影响。
"""

import trueskill
import random
from typing import List, Tuple, Dict


def calculate_trueskill_scores(
    edges: List[Tuple[str, str]],
    n_shuffles: int = 5,
    mu: float = 25.0,
    sigma: float = 8.333,
    beta: float = 4.166,
    tau: float = 0.083,
    seed: int = None
) -> Dict[str, float]:
    """
    从有向图边表计算每个节点的 TrueSkill 分数。
    
    Args:
        edges: 有向图的边表，每个 tuple 为 (winner, loser)，表示 winner 胜过 loser
        n_shuffles: shuffle 次数，用于消除输入顺序的影响，默认 5 次
        mu: TrueSkill 初始分数均值，默认 25.0
        sigma: TrueSkill 初始不确定度，默认 8.333
        beta: TrueSkill beta 参数，默认 4.166
        tau: TrueSkill tau 参数，默认 0.083
        seed: 随机种子，用于复现结果，默认 None
        
    Returns:
        Dict[str, float]: nodename -> score 的字典，score 为各次 shuffle 计算后 mu 的平均值
        
    Example:
        >>> edges = [("A", "B"), ("B", "C"), ("A", "C")]  # A > B, B > C, A > C
        >>> scores = calculate_trueskill_scores(edges)
        >>> # scores["A"] > scores["B"] > scores["C"]
    """
    if seed is not None:
        random.seed(seed)
    
    # 处理空边表
    if not edges:
        return {}
    
    # 收集所有节点
    all_nodes = set()
    for winner, loser in edges:
        all_nodes.add(winner)
        all_nodes.add(loser)
    
    # 存储每次 shuffle 后的分数
    all_scores = {node: [] for node in all_nodes}
    
    for _ in range(n_shuffles):
        # Shuffle edges
        shuffled_edges = list(edges)
        random.shuffle(shuffled_edges)
        
        # 创建 TrueSkill 环境
        env = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau)
        ratings = {node: env.create_rating() for node in all_nodes}
        
        # 处理所有边
        for winner, loser in shuffled_edges:
            rating_winner = ratings[winner]
            rating_loser = ratings[loser]
            
            # rate 返回排名顺序的 rating，第一个是赢家
            new_rating_winner, new_rating_loser = env.rate(
                [[rating_winner], [rating_loser]]
            )
            ratings[winner] = new_rating_winner[0]
            ratings[loser] = new_rating_loser[0]
        
        # 收集分数
        for node in all_nodes:
            all_scores[node].append(ratings[node].mu)
    
    # 计算平均分数
    result = {node: sum(scores) / len(scores) for node, scores in all_scores.items()}
    
    return result


if __name__ == "__main__":
    # 简单示例
    edges = [
        ("Alice", "Bob"),    # Alice 胜 Bob
        ("Bob", "Charlie"),  # Bob 胜 Charlie
        ("Alice", "Charlie"),# Alice 胜 Charlie
    ]
    
    scores = calculate_trueskill_scores(edges, seed=42)
    
    # 按分数排序输出
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("TrueSkill 分数（从高到低）：")
    for name, score in sorted_scores:
        print(f"  {name}: {score:.2f}")

