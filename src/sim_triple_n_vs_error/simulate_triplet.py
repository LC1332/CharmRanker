"""
模拟Triplet比较方法在不同API调用次数下的rank error。

模拟设置：
- N=200个样本，latent score在[1,100]均匀分布
- Triplet比较：每次比较3个物体，标出最大和最小
- 使用Plackett-Luce模型建模triplet排序概率
- 一次triplet比较产生3条边
- 研究K'=2,4,8,16,32次平均API调用下的rank error
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from tqdm import tqdm

# 添加项目根目录到path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trueskill.calculate_trueskill_scores import calculate_trueskill_scores


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
                        = [exp(k*s_a) / sum(exp(k*s_*))] * [exp(k*s_b) / (exp(k*s_b) + exp(k*s_c))]
    
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
    
    # 使用Plackett-Luce模型逐步选择
    for _ in range(2):  # 选择第1名和第2名，第3名自动确定
        # 计算每个候选的概率
        exp_scores = np.exp(logistic_k * np.array(remaining_scores))
        probs = exp_scores / exp_scores.sum()
        
        # 随机选择
        choice_idx = rng.choice(len(remaining_scores), p=probs)
        result.append(remaining_indices[choice_idx])
        
        # 移除已选择的
        remaining_indices.pop(choice_idx)
        remaining_scores.pop(choice_idx)
    
    # 第3名
    result.append(remaining_indices[0])
    
    return tuple(result)  # (first, second, third) 表示 first > second > third


def triplet_to_edges(first: int, second: int, third: int, item_ids: list) -> list:
    """
    将triplet排序结果转换为边表。
    
    Args:
        first, second, third: 排序后的原始索引
        item_ids: item ID列表
        
    Returns:
        [(winner, loser), ...] 3条边
    """
    return [
        (item_ids[first], item_ids[second]),   # first > second
        (item_ids[first], item_ids[third]),    # first > third
        (item_ids[second], item_ids[third]),   # second > third
    ]


def generate_triplets(n_items: int, n_triplets: int, rng: np.random.Generator) -> list:
    """
    生成triplet比较的三元组。
    
    Args:
        n_items: item数量
        n_triplets: 需要生成的triplet数量
        rng: 随机数生成器
        
    Returns:
        [(i, j, k), ...] 三元组列表
    """
    triplets = set()
    
    while len(triplets) < n_triplets:
        # 随机选择3个不同的item
        selected = tuple(sorted(rng.choice(n_items, size=3, replace=False)))
        triplets.add(selected)
    
    return list(triplets)


def run_simulation(
    n_items: int = 200,
    k_prime_values: list = None,
    n_trials: int = 10,
    seed: int = 42
) -> dict:
    """
    运行triplet比较的模拟实验。
    
    Args:
        n_items: item数量
        k_prime_values: 每个item平均API调用次数列表
        n_trials: 每个k'值重复实验次数
        seed: 随机种子
        
    Returns:
        {k': {'mean': mean_normalized_rank_error, 'std': std}}
    """
    if k_prime_values is None:
        k_prime_values = [2, 4, 8, 16, 32]
    
    rng = np.random.default_rng(seed)
    logistic_k = compute_logistic_k()
    
    print(f"Triplet比较模拟")
    print(f"Logistic参数 k = {logistic_k:.6f}")
    print(f"N = {n_items}, 重复实验 {n_trials} 次")
    print("-" * 50)
    
    results = {}
    
    for k_prime in tqdm(k_prime_values, desc="K' values", position=0):
        trial_errors = []
        
        # 计算triplet数量：K'N次triplet比较
        n_triplets = k_prime * n_items
        
        for trial in tqdm(range(n_trials), desc=f"K'={k_prime} trials", position=1, leave=False):
            # 生成latent scores
            latent_scores = rng.uniform(1, 100, n_items)
            item_ids = [str(i) for i in range(n_items)]
            
            # 计算ground truth rank (从高到低)
            gt_rank_order = np.argsort(-latent_scores)
            gt_rank = np.zeros(n_items, dtype=int)
            for rank, idx in enumerate(gt_rank_order):
                gt_rank[idx] = rank
            
            # 生成triplets
            triplets = generate_triplets(n_items, n_triplets, rng)
            
            # 进行triplet比较并生成边表
            edges = []
            for i, j, k in triplets:
                # 进行triplet比较
                first, second, third = triplet_comparison(
                    latent_scores[i], latent_scores[j], latent_scores[k],
                    logistic_k, rng
                )
                # 将相对索引转换为原始索引
                original_indices = [i, j, k]
                first_orig = original_indices[first]
                second_orig = original_indices[second]
                third_orig = original_indices[third]
                
                # 生成3条边
                edges.extend(triplet_to_edges(first_orig, second_orig, third_orig, item_ids))
            
            # 计算TrueSkill分数
            ts_scores = calculate_trueskill_scores(edges, n_shuffles=5, seed=trial)
            
            # 计算TrueSkill rank
            ts_rank_list = sorted(ts_scores.items(), key=lambda x: x[1], reverse=True)
            ts_rank = {item_id: rank for rank, (item_id, _) in enumerate(ts_rank_list)}
            
            # 计算rank error
            rank_errors = []
            for i in range(n_items):
                item_id = str(i)
                if item_id in ts_rank:
                    error = abs(gt_rank[i] - ts_rank[item_id])
                    rank_errors.append(error)
            
            mean_error = np.mean(rank_errors)
            trial_errors.append(mean_error)
        
        mean_normalized_error = np.mean(trial_errors) / n_items
        std_normalized_error = np.std(trial_errors) / n_items
        
        results[k_prime] = {
            'mean': mean_normalized_error,
            'std': std_normalized_error,
            'n_edges': 3 * k_prime * n_items  # 总边数
        }
        
        tqdm.write(f"  K'={k_prime}: rank_error/N = {mean_normalized_error:.4f} ± {std_normalized_error:.4f} (edges: {3*k_prime*n_items})")
    
    return results


def plot_results(results: dict, output_path: Path):
    """
    绘制K' vs rank_error/N的图。
    """
    k_values = sorted(results.keys())
    means = [results[k]['mean'] for k in k_values]
    stds = [results[k]['std'] for k in k_values]
    
    plt.figure(figsize=(10, 6))
    
    plt.errorbar(k_values, means, yerr=stds, fmt='o-', capsize=5, 
                 linewidth=2, markersize=8, color='#E94F37', 
                 ecolor='#1C77C3', capthick=2)
    
    plt.xlabel("K' (Average API Calls per Item)", fontsize=12)
    plt.ylabel('Normalized Rank Error (rank_error / N)', fontsize=12)
    plt.title("Triplet Comparison: Rank Error vs API Calls\n(N=200 items, Plackett-Luce model, 3 edges per triplet)", 
              fontsize=14)
    
    plt.xscale('log', base=2)
    plt.xticks(k_values, [str(k) for k in k_values])
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到: {output_path}")
    plt.close()


def plot_comparison(triplet_results: dict, output_path: Path):
    """
    绘制triplet和pairwise方法的对比图（按API调用次数对齐）。
    """
    # 加载之前pairwise的结果（如果存在的话，这里重新计算）
    from src.sim_trueskill_n_vs_error.simulate_trueskill import run_simulation as run_pairwise
    
    print("\n运行pairwise对比实验...")
    # pairwise: K次比较 = K*N/2 次API调用
    # 所以K' = K/2 次API调用对应 K次比较
    # K' = 2,4,8,16,32 对应 K = 4,8,16,32,64
    pairwise_results = run_pairwise(
        n_items=200,
        k_values=[4, 8, 16, 32, 64],
        n_trials=10,
        seed=42
    )
    
    # 转换为按API调用次数
    # pairwise: K次比较 -> K/2次API调用
    pairwise_by_api = {k//2: v for k, v in pairwise_results.items()}
    
    plt.figure(figsize=(10, 6))
    
    # Triplet结果
    k_triplet = sorted(triplet_results.keys())
    means_triplet = [triplet_results[k]['mean'] for k in k_triplet]
    stds_triplet = [triplet_results[k]['std'] for k in k_triplet]
    
    plt.errorbar(k_triplet, means_triplet, yerr=stds_triplet, fmt='o-', capsize=5,
                 linewidth=2, markersize=8, color='#E94F37', 
                 ecolor='#E94F37', capthick=2, alpha=0.8, label='Triplet (3 edges/call)')
    
    # Pairwise结果
    k_pairwise = sorted(pairwise_by_api.keys())
    means_pairwise = [pairwise_by_api[k]['mean'] for k in k_pairwise]
    stds_pairwise = [pairwise_by_api[k]['std'] for k in k_pairwise]
    
    plt.errorbar(k_pairwise, means_pairwise, yerr=stds_pairwise, fmt='s--', capsize=5,
                 linewidth=2, markersize=8, color='#2E86AB', 
                 ecolor='#2E86AB', capthick=2, alpha=0.8, label='Pairwise (1 edge/call)')
    
    plt.xlabel("K' (Average API Calls per Item)", fontsize=12)
    plt.ylabel('Normalized Rank Error (rank_error / N)', fontsize=12)
    plt.title("Triplet vs Pairwise: Rank Error vs API Calls\n(N=200 items, same API budget)", fontsize=14)
    
    plt.xscale('log', base=2)
    plt.xticks(k_triplet, [str(k) for k in k_triplet])
    
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存到: {output_path}")
    plt.close()


def main():
    """主函数"""
    # 运行triplet模拟
    triplet_results = run_simulation(
        n_items=200,
        k_prime_values=[2, 4, 8, 16, 32],
        n_trials=10,
        seed=42
    )
    
    # 绘图并保存
    output_dir = project_root / "local_data" / "visualization" / "trueskill_simu"
    
    # Triplet单独的图
    triplet_path = output_dir / "triplet_k_vs_rank_error.jpg"
    plot_results(triplet_results, triplet_path)
    
    # Triplet vs Pairwise对比图
    comparison_path = output_dir / "triplet_vs_pairwise_comparison.jpg"
    plot_comparison(triplet_results, comparison_path)
    
    print("\n模拟完成!")


if __name__ == "__main__":
    main()

