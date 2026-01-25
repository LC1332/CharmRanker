"""
模拟不同的triplet比较策略在固定API budget下的rank error表现。

设置：
- N = 5000 个样本
- Budget = 30000 次API调用
- 每个item的latent score在[1,100]均匀分布

方案：
1. 平均比较：每个样本调用相同次数
2. Anchor策略（N_anchor=200）
3. Anchor策略（N_anchor=500）
4. Anchor策略（N_anchor=1000）
5. 动态调整（threshold=20）
6. 动态调整（threshold=40）
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm
from typing import List, Tuple, Dict
from dataclasses import dataclass

# 添加项目根目录到path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trueskill.calculate_trueskill_scores import calculate_trueskill_scores


# ============= 通用工具函数 =============

def compute_logistic_k():
    """
    计算logistic函数的参数k。
    与pairwise比较保持一致：latent差=10时分对概率73%
    """
    k = -np.log(1/0.73 - 1) / 10
    return k


def triplet_comparison(
    latent_i: float, 
    latent_j: float, 
    latent_k: float, 
    logistic_k: float, 
    rng: np.random.Generator
) -> tuple:
    """
    对三个物体进行triplet比较，返回完整排序。
    
    使用Plackett-Luce模型：
    P(排序为 a > b > c) = P(a是最大) * P(b是次大|a已选)
    
    Args:
        latent_i, latent_j, latent_k: 三个物体的latent scores
        logistic_k: logistic函数参数
        rng: 随机数生成器
        
    Returns:
        (first, second, third): 排序后的索引，first > second > third
    """
    scores = np.array([latent_i, latent_j, latent_k])
    indices = np.array([0, 1, 2])
    
    result = []
    remaining_indices = list(range(3))
    remaining_scores = list(scores)
    
    for _ in range(2):
        exp_scores = np.exp(logistic_k * np.array(remaining_scores))
        probs = exp_scores / exp_scores.sum()
        
        choice_idx = rng.choice(len(remaining_scores), p=probs)
        result.append(remaining_indices[choice_idx])
        
        remaining_indices.pop(choice_idx)
        remaining_scores.pop(choice_idx)
    
    result.append(remaining_indices[0])
    
    return tuple(result)


def triplet_to_edges(first: int, second: int, third: int, item_ids: list) -> list:
    """
    将triplet排序结果转换为边表。
    """
    return [
        (item_ids[first], item_ids[second]),
        (item_ids[first], item_ids[third]),
        (item_ids[second], item_ids[third]),
    ]


def perform_triplet_comparison(
    i: int, j: int, k: int,
    latent_scores: np.ndarray,
    item_ids: list,
    logistic_k: float,
    rng: np.random.Generator
) -> list:
    """
    执行一次triplet比较并返回边。
    """
    first, second, third = triplet_comparison(
        latent_scores[i], latent_scores[j], latent_scores[k],
        logistic_k, rng
    )
    original_indices = [i, j, k]
    first_orig = original_indices[first]
    second_orig = original_indices[second]
    third_orig = original_indices[third]
    
    return triplet_to_edges(first_orig, second_orig, third_orig, item_ids)


def calculate_rank_error(
    latent_scores: np.ndarray,
    edges: list,
    item_ids: list,
    n_items: int
) -> float:
    """
    计算normalized rank error。
    """
    # Ground truth rank
    gt_rank_order = np.argsort(-latent_scores)
    gt_rank = np.zeros(n_items, dtype=int)
    for rank, idx in enumerate(gt_rank_order):
        gt_rank[idx] = rank
    
    # TrueSkill rank
    ts_scores = calculate_trueskill_scores(edges, n_shuffles=5, seed=42)
    ts_rank_list = sorted(ts_scores.items(), key=lambda x: x[1], reverse=True)
    ts_rank = {item_id: rank for rank, (item_id, _) in enumerate(ts_rank_list)}
    
    # Rank error
    rank_errors = []
    for i in range(n_items):
        item_id = str(i)
        if item_id in ts_rank:
            error = abs(gt_rank[i] - ts_rank[item_id])
            rank_errors.append(error)
        else:
            rank_errors.append(n_items)
    
    return np.mean(rank_errors) / n_items


# ============= 方案1：平均比较 =============

def plan1_uniform(
    n_items: int,
    budget: int,
    latent_scores: np.ndarray,
    item_ids: list,
    logistic_k: float,
    rng: np.random.Generator
) -> list:
    """
    方案1：平均比较，每个样本调用相同次数。
    
    Returns:
        edges: 所有比较产生的边
    """
    edges = []
    
    for _ in tqdm(range(budget), desc="方案1-平均比较"):
        # 随机选择3个不同的item
        selected = rng.choice(n_items, size=3, replace=False)
        i, j, k = selected
        
        new_edges = perform_triplet_comparison(
            i, j, k, latent_scores, item_ids, logistic_k, rng
        )
        edges.extend(new_edges)
    
    return edges


# ============= 方案2-4：Anchor策略 =============

def plan_anchor(
    n_items: int,
    budget: int,
    n_anchor: int,
    latent_scores: np.ndarray,
    item_ids: list,
    logistic_k: float,
    rng: np.random.Generator,
    plan_name: str = "Anchor"
) -> list:
    """
    Anchor策略：将样本划分为Anchor set和Pool set。
    每次以(Anchor, Anchor, Pool)的组合进行比较。
    
    Args:
        n_anchor: Anchor set大小
        
    Returns:
        edges: 所有比较产生的边
    """
    edges = []
    
    # 随机划分Anchor和Pool
    all_indices = np.arange(n_items)
    rng.shuffle(all_indices)
    anchor_indices = all_indices[:n_anchor]
    pool_indices = all_indices[n_anchor:]
    
    for _ in tqdm(range(budget), desc=f"{plan_name}"):
        # 选择2个不同的anchor和1个pool
        selected_anchors = rng.choice(anchor_indices, size=2, replace=False)
        selected_pool = rng.choice(pool_indices, size=1)
        
        i, j = selected_anchors
        k = selected_pool[0]
        
        new_edges = perform_triplet_comparison(
            i, j, k, latent_scores, item_ids, logistic_k, rng
        )
        edges.extend(new_edges)
    
    return edges


# ============= 方案5-6：动态调整策略 =============

def plan_dynamic(
    n_items: int,
    budget: int,
    threshold: int,
    latent_scores: np.ndarray,
    item_ids: list,
    logistic_k: float,
    rng: np.random.Generator,
    initial_budget: int = 10000,
    batch_size: int = 500,
    plan_name: str = "动态调整"
) -> list:
    """
    动态调整策略：
    1. 先进行initial_budget次API调用
    2. 计算分数并归一化到0-100
    3. 之后每次选batch_size个triplet，三个item的最大差保持在threshold以内
    4. 循环直到budget用完
    
    Args:
        threshold: triplet中item分数最大差的阈值
        initial_budget: 初始API调用次数
        batch_size: 每批triplet数量
        
    Returns:
        edges: 所有比较产生的边
    """
    edges = []
    remaining_budget = budget
    
    # 第一阶段：随机比较
    print(f"  阶段1: 随机比较 {initial_budget} 次")
    for _ in tqdm(range(initial_budget), desc=f"{plan_name}-阶段1"):
        selected = rng.choice(n_items, size=3, replace=False)
        i, j, k = selected
        
        new_edges = perform_triplet_comparison(
            i, j, k, latent_scores, item_ids, logistic_k, rng
        )
        edges.extend(new_edges)
    
    remaining_budget -= initial_budget
    
    # 计算当前分数
    current_scores = calculate_trueskill_scores(edges, n_shuffles=5, seed=42)
    
    # 将分数根据rank归一化到0-100
    def normalize_scores(scores_dict):
        if not scores_dict:
            return {str(i): 50 for i in range(n_items)}
        
        sorted_items = sorted(scores_dict.items(), key=lambda x: x[1])
        n = len(sorted_items)
        normalized = {}
        for rank, (item_id, _) in enumerate(sorted_items):
            normalized[item_id] = (rank / max(n-1, 1)) * 100
        return normalized
    
    normalized_scores = normalize_scores(current_scores)
    
    # 确保所有item都有分数
    for i in range(n_items):
        item_id = str(i)
        if item_id not in normalized_scores:
            normalized_scores[item_id] = 50
    
    # 第二阶段：基于阈值的比较
    print(f"  阶段2: 阈值比较 (threshold={threshold})")
    pbar = tqdm(total=remaining_budget, desc=f"{plan_name}-阶段2")
    
    while remaining_budget > 0:
        # 尝试选择满足阈值条件的triplet
        batch = min(batch_size, remaining_budget)
        valid_triplets = 0
        max_attempts = batch * 100  # 防止无限循环
        attempts = 0
        
        while valid_triplets < batch and attempts < max_attempts:
            attempts += 1
            
            # 随机选择3个不同的item
            selected = rng.choice(n_items, size=3, replace=False)
            i, j, k = selected
            
            # 检查分数差是否在阈值内
            scores = [normalized_scores[str(i)], 
                     normalized_scores[str(j)], 
                     normalized_scores[str(k)]]
            max_diff = max(scores) - min(scores)
            
            if max_diff <= threshold:
                new_edges = perform_triplet_comparison(
                    i, j, k, latent_scores, item_ids, logistic_k, rng
                )
                edges.extend(new_edges)
                valid_triplets += 1
                pbar.update(1)
        
        remaining_budget -= valid_triplets
        
        if valid_triplets < batch:
            print(f"\n  警告: 只找到 {valid_triplets}/{batch} 个满足阈值的triplet")
        
        # 更新分数
        if remaining_budget > 0:
            current_scores = calculate_trueskill_scores(edges, n_shuffles=5, seed=42)
            normalized_scores = normalize_scores(current_scores)
            for i in range(n_items):
                item_id = str(i)
                if item_id not in normalized_scores:
                    normalized_scores[item_id] = 50
    
    pbar.close()
    return edges


# ============= 主要模拟函数 =============

@dataclass
class SimulationResult:
    plan_name: str
    rank_error: float
    n_edges: int


def run_all_plans(
    n_items: int = 5000,
    budget: int = 30000,
    n_trials: int = 3,
    seed: int = 42
) -> Dict[str, Dict]:
    """
    运行所有方案的模拟实验。
    
    Args:
        n_items: item数量
        budget: API调用预算
        n_trials: 重复实验次数
        seed: 随机种子
        
    Returns:
        results: {plan_name: {'mean': mean_error, 'std': std_error, 'trials': [...]}}
    """
    logistic_k = compute_logistic_k()
    
    print(f"模拟设置:")
    print(f"  N = {n_items} 个样本")
    print(f"  Budget = {budget} 次API调用")
    print(f"  重复实验 {n_trials} 次")
    print(f"  Logistic k = {logistic_k:.6f}")
    print("=" * 60)
    
    plans = [
        ("方案1-平均比较", lambda ls, ids, lk, rng: plan1_uniform(n_items, budget, ls, ids, lk, rng)),
        ("方案2-Anchor200", lambda ls, ids, lk, rng: plan_anchor(n_items, budget, 200, ls, ids, lk, rng, "方案2-Anchor200")),
        ("方案3-Anchor500", lambda ls, ids, lk, rng: plan_anchor(n_items, budget, 500, ls, ids, lk, rng, "方案3-Anchor500")),
        ("方案4-Anchor1000", lambda ls, ids, lk, rng: plan_anchor(n_items, budget, 1000, ls, ids, lk, rng, "方案4-Anchor1000")),
        ("方案5-动态T20", lambda ls, ids, lk, rng: plan_dynamic(n_items, budget, 20, ls, ids, lk, rng, plan_name="方案5-动态T20")),
        ("方案6-动态T40", lambda ls, ids, lk, rng: plan_dynamic(n_items, budget, 40, ls, ids, lk, rng, plan_name="方案6-动态T40")),
    ]
    
    results = {}
    
    for plan_name, plan_func in plans:
        print(f"\n{'='*60}")
        print(f"运行 {plan_name}")
        print("=" * 60)
        
        trial_errors = []
        
        for trial in range(n_trials):
            print(f"\n--- Trial {trial+1}/{n_trials} ---")
            
            # 每次trial使用不同的随机种子
            trial_seed = seed + trial * 1000
            rng = np.random.default_rng(trial_seed)
            
            # 生成latent scores
            latent_scores = rng.uniform(1, 100, n_items)
            item_ids = [str(i) for i in range(n_items)]
            
            # 运行方案
            edges = plan_func(latent_scores, item_ids, logistic_k, rng)
            
            # 计算rank error
            error = calculate_rank_error(latent_scores, edges, item_ids, n_items)
            trial_errors.append(error)
            
            print(f"  Rank Error/N = {error:.4f}")
        
        mean_error = np.mean(trial_errors)
        std_error = np.std(trial_errors)
        
        results[plan_name] = {
            'mean': mean_error,
            'std': std_error,
            'trials': trial_errors
        }
        
        print(f"\n{plan_name} 结果: {mean_error:.4f} ± {std_error:.4f}")
    
    return results


def plot_results(results: Dict[str, Dict], output_path: Path):
    """
    绘制各方案对比图。
    """
    plt.figure(figsize=(12, 6))
    
    plan_names = list(results.keys())
    means = [results[p]['mean'] for p in plan_names]
    stds = [results[p]['std'] for p in plan_names]
    
    # 使用不同颜色
    colors = ['#E94F37', '#1C77C3', '#39A78E', '#8B4513', '#9B59B6', '#F39C12']
    
    bars = plt.bar(range(len(plan_names)), means, yerr=stds, 
                   capsize=5, color=colors[:len(plan_names)], alpha=0.8)
    
    plt.xlabel('方案', fontsize=12)
    plt.ylabel('Normalized Rank Error (rank_error / N)', fontsize=12)
    plt.title('不同比较策略的Rank Error对比\n(N=5000, Budget=30000)', fontsize=14)
    
    plt.xticks(range(len(plan_names)), 
               [p.split('-')[1] for p in plan_names], 
               rotation=45, ha='right')
    
    # 在柱子上方显示数值
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.002,
                f'{mean:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    plt.close()


def print_summary(results: Dict[str, Dict]):
    """
    打印结果汇总表。
    """
    print("\n" + "=" * 60)
    print("结果汇总")
    print("=" * 60)
    print(f"{'方案':<20} {'Mean Error':<15} {'Std':<15}")
    print("-" * 50)
    
    # 按mean error排序
    sorted_plans = sorted(results.items(), key=lambda x: x[1]['mean'])
    
    for plan_name, data in sorted_plans:
        print(f"{plan_name:<20} {data['mean']:.4f}         {data['std']:.4f}")
    
    print("-" * 50)
    print(f"最佳方案: {sorted_plans[0][0]} (Error={sorted_plans[0][1]['mean']:.4f})")


def main():
    """主函数"""
    # 运行所有方案
    results = run_all_plans(
        n_items=5000,
        budget=30000,
        n_trials=3,
        seed=42
    )
    
    # 打印汇总
    print_summary(results)
    
    # 绘制对比图
    output_dir = project_root / "local_data" / "visualization" / "different_plans"
    output_path = output_dir / "plan_comparison.jpg"
    plot_results(results, output_path)
    
    print("\n模拟完成!")


if __name__ == "__main__":
    main()

