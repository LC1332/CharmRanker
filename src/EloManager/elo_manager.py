"""
EloManager - 管理节点和边的信息，提出triplet用于标注

主要功能:
- save/load: 保存和加载节点及比较日志
- add_nodes: 添加节点
- propose_triplet: 根据规则提出K个triplet
- submit_triplet_result: 提交triplet比较结果
"""

import json
import hashlib
import yaml
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from itertools import combinations
import sys

# 添加项目根目录到path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trueskill.calculate_true_skill import TrueSkillCalculator


class EloManager:
    """
    管理节点和triplet比较的类。
    
    使用TrueSkill算法进行分数计算，支持增量更新。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化EloManager。
        
        Args:
            config_path: 配置文件路径，默认使用同目录下的config.yaml
        """
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        self.config = self._load_config(config_path)
        
        # 初始化TrueSkill计算器
        ts_config = self.config['trueskill']
        self.calculator = TrueSkillCalculator(
            mu=ts_config['mu'],
            sigma=ts_config['sigma'],
            beta=ts_config['beta'],
            tau=ts_config['tau']
        )
        
        # 节点信息: {name: node_info}
        self.nodes: Dict[str, Dict[str, Any]] = {}
        
        # 比较日志: 存储所有triplet比较结果
        self.comparison_logs: List[Dict[str, Any]] = []
        
        # 已比较的triplet集合 (用frozenset存储，不考虑顺序)
        self.compared_triplets: Set[frozenset] = set()
        
        # 每个节点的参与次数
        self.node_compare_counts: Dict[str, int] = {}
        
        # Propose配置
        propose_config = self.config['propose']
        self.minimal_compare_times = propose_config['minimal_compare_times']
        self.rank_threshold = propose_config['rank_threshold']
        self.beauty_weight = propose_config['beauty_weight']
        self.candidate_multiplier = propose_config['candidate_multiplier']
        
        # 随机数生成器
        self.rng = np.random.default_rng(self.config['simulation']['seed'])
    
    def _load_config(self, config_path: str) -> dict:
        """加载配置文件"""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _generate_hash_name(self, node_info: dict) -> str:
        """为节点生成hash名称"""
        info_str = json.dumps(node_info, sort_keys=True)
        return hashlib.md5(info_str.encode()).hexdigest()[:16]
    
    def save(self, path: str) -> None:
        """
        保存节点和比较日志信息到指定路径。
        
        Args:
            path: 保存路径（目录）
        """
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取最终分数
        final_scores = self.calculator.get_final_scores()
        
        # 保存节点信息（包含score）
        nodes_data = []
        for name, info in self.nodes.items():
            node_data = info.copy()
            node_data['name'] = name
            if name in final_scores:
                node_data['score'] = {
                    'mu': final_scores[name]['mu'],
                    'sigma': final_scores[name]['sigma']
                }
            else:
                # 未参与比较的节点使用默认分数
                node_data['score'] = {
                    'mu': self.config['trueskill']['mu'],
                    'sigma': self.config['trueskill']['sigma']
                }
            nodes_data.append(node_data)
        
        nodes_path = save_dir / "nodes.json"
        with open(nodes_path, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, ensure_ascii=False, indent=2)
        
        # 保存比较日志
        logs_path = save_dir / "comparison_logs.jsonl"
        with open(logs_path, 'w', encoding='utf-8') as f:
            for log in self.comparison_logs:
                f.write(json.dumps(log, ensure_ascii=False) + '\n')
        
        # 保存元数据
        meta = {
            'total_nodes': len(self.nodes),
            'total_comparisons': len(self.comparison_logs),
            'node_compare_counts': self.node_compare_counts
        }
        meta_path = save_dir / "meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        
        print(f"已保存到 {save_dir}: {len(self.nodes)} 个节点, {len(self.comparison_logs)} 条比较记录")
    
    def load(self, path: str) -> None:
        """
        从指定路径加载节点和比较日志信息。
        
        Args:
            path: 加载路径（目录）
        """
        load_dir = Path(path)
        if not load_dir.exists():
            raise FileNotFoundError(f"路径不存在: {load_dir}")
        
        # 加载节点信息
        nodes_path = load_dir / "nodes.json"
        if nodes_path.exists():
            with open(nodes_path, 'r', encoding='utf-8') as f:
                nodes_data = json.load(f)
            
            for node_data in nodes_data:
                name = node_data.pop('name')
                score = node_data.pop('score', None)
                self.nodes[name] = node_data
                
                # 恢复TrueSkill rating
                if score:
                    self.calculator.ratings[name] = self.calculator.env.Rating(
                        mu=score['mu'], 
                        sigma=score['sigma']
                    )
        
        # 加载比较日志
        logs_path = load_dir / "comparison_logs.jsonl"
        if logs_path.exists():
            with open(logs_path, 'r', encoding='utf-8') as f:
                for line in f:
                    log = json.loads(line.strip())
                    self.comparison_logs.append(log)
                    
                    # 重建已比较triplet集合
                    triplet = frozenset(log['triplet'])
                    self.compared_triplets.add(triplet)
        
        # 加载元数据
        meta_path = load_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            self.node_compare_counts = meta.get('node_compare_counts', {})
        else:
            # 从比较日志重建参与次数
            self._rebuild_compare_counts()
        
        print(f"已从 {load_dir} 加载: {len(self.nodes)} 个节点, {len(self.comparison_logs)} 条比较记录")
    
    def _rebuild_compare_counts(self) -> None:
        """从比较日志重建节点参与次数"""
        self.node_compare_counts = {name: 0 for name in self.nodes}
        for log in self.comparison_logs:
            for node_name in log['triplet']:
                self.node_compare_counts[node_name] = self.node_compare_counts.get(node_name, 0) + 1
    
    def add_nodes(self, node_infos: List[Dict[str, Any]]) -> List[str]:
        """
        添加节点。
        
        Args:
            node_infos: 节点信息列表，每个元素是一个dict
            
        Returns:
            添加的节点名称列表
        """
        added_names = []
        
        for info in node_infos:
            info = info.copy()
            
            # 获取或生成name
            if 'name' in info:
                name = info['name']
            else:
                name = self._generate_hash_name(info)
                info['name'] = name
            
            # 检查是否重复
            if name in self.nodes:
                print(f"警告: 节点 {name} 已存在，跳过")
                continue
            
            # 添加节点
            self.nodes[name] = info
            self.node_compare_counts[name] = 0
            added_names.append(name)
        
        print(f"添加了 {len(added_names)} 个节点，当前共 {len(self.nodes)} 个节点")
        return added_names
    
    def _get_normalized_scores(self) -> Dict[str, float]:
        """
        获取归一化到0-100的分数（基于rank）。
        
        Returns:
            {node_name: normalized_score}
        """
        final_scores = self.calculator.get_final_scores()
        
        if not final_scores:
            # 没有任何比较，所有节点给50分
            return {name: 50.0 for name in self.nodes}
        
        # 按mu排序
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1]['mu'])
        n = len(sorted_items)
        
        normalized = {}
        for rank, (name, _) in enumerate(sorted_items):
            normalized[name] = (rank / max(n - 1, 1)) * 100
        
        # 未参与比较的节点给50分
        for name in self.nodes:
            if name not in normalized:
                normalized[name] = 50.0
        
        return normalized
    
    def _get_hungry_nodes(self) -> List[str]:
        """获取未温饱的节点（参与次数 < minimal_compare_times）"""
        return [
            name for name, count in self.node_compare_counts.items()
            if count < self.minimal_compare_times
        ]
    
    def _is_triplet_compared(self, triplet: Tuple[str, str, str]) -> bool:
        """检查triplet是否已被比较过"""
        return frozenset(triplet) in self.compared_triplets
    
    def _sample_hungry_triplets(self, K: int) -> List[Tuple[str, str, str]]:
        """
        温饱阶段的triplet采样。
        优先让未温饱节点和未温饱节点比较。
        
        Args:
            K: 需要采样的triplet数量
            
        Returns:
            triplet列表
        """
        hungry_nodes = self._get_hungry_nodes()
        all_nodes = list(self.nodes.keys())
        
        if len(hungry_nodes) < 3:
            # 未温饱节点不足3个，混合采样
            if len(hungry_nodes) == 0:
                return []
            
            # 确保至少有一个未温饱节点参与
            candidates = []
            for _ in range(K * 10):  # 多尝试几次
                if len(hungry_nodes) >= 1:
                    # 选1-2个未温饱节点 + 其他节点
                    n_hungry = min(len(hungry_nodes), self.rng.integers(1, 4))
                    selected_hungry = self.rng.choice(hungry_nodes, size=n_hungry, replace=False).tolist()
                    
                    remaining = [n for n in all_nodes if n not in selected_hungry]
                    n_remaining = 3 - n_hungry
                    if len(remaining) >= n_remaining:
                        selected_other = self.rng.choice(remaining, size=n_remaining, replace=False).tolist()
                        triplet = tuple(selected_hungry + selected_other)
                        
                        if not self._is_triplet_compared(triplet):
                            candidates.append(triplet)
                            if len(candidates) >= K:
                                break
            
            return candidates[:K]
        
        # 未温饱节点足够，优先从中采样
        triplets = []
        attempts = 0
        max_attempts = K * 100
        
        while len(triplets) < K and attempts < max_attempts:
            attempts += 1
            
            # 优先全部从未温饱节点中选
            if len(hungry_nodes) >= 3:
                selected = self.rng.choice(hungry_nodes, size=3, replace=False)
            else:
                # 混合选择
                n_hungry = len(hungry_nodes)
                selected_hungry = self.rng.choice(hungry_nodes, size=n_hungry, replace=False).tolist()
                remaining = [n for n in all_nodes if n not in selected_hungry]
                selected_other = self.rng.choice(remaining, size=3-n_hungry, replace=False).tolist()
                selected = selected_hungry + selected_other
            
            triplet = tuple(selected)
            
            if not self._is_triplet_compared(triplet):
                triplets.append(triplet)
        
        return triplets
    
    def _sample_nearby_triplets(self, K: int) -> List[Tuple[str, str, str]]:
        """
        邻近抽样 + 高分多抽。
        
        Args:
            K: 需要采样的triplet数量
            
        Returns:
            triplet列表
        """
        normalized_scores = self._get_normalized_scores()
        all_nodes = list(self.nodes.keys())
        
        if len(all_nodes) < 3:
            return []
        
        # 构造2K个候选triplet
        n_candidates = K * self.candidate_multiplier
        candidates = []
        attempts = 0
        max_attempts = n_candidates * 100
        
        while len(candidates) < n_candidates and attempts < max_attempts:
            attempts += 1
            
            # 随机选3个节点
            selected = self.rng.choice(all_nodes, size=3, replace=False)
            triplet = tuple(selected)
            
            # 检查是否已比较
            if self._is_triplet_compared(triplet):
                continue
            
            # 检查邻近原则
            scores = [normalized_scores[n] for n in triplet]
            max_diff = max(scores) - min(scores)
            
            if max_diff > self.rank_threshold:
                continue
            
            candidates.append((triplet, scores))
        
        if not candidates:
            # 放宽阈值，确保能返回结果
            print("警告: 无法找到满足邻近原则的triplet，放宽阈值")
            return self._sample_fallback_triplets(K)
        
        # 高分多抽原则：计算调和平均数并加权
        def compute_weight(scores: List[float]) -> float:
            """计算权重：调和平均数接近0权重为1，接近100权重为beauty_weight"""
            harmonic_mean = 3 / (1/(scores[0]+1) + 1/(scores[1]+1) + 1/(scores[2]+1))
            # 线性插值权重
            weight = 1 + (self.beauty_weight - 1) * (harmonic_mean / 100)
            return weight
        
        weights = np.array([compute_weight(scores) for triplet, scores in candidates])
        weights = weights / weights.sum()  # 归一化
        
        # 加权采样
        if len(candidates) <= K:
            return [triplet for triplet, _ in candidates]
        
        indices = self.rng.choice(len(candidates), size=K, replace=False, p=weights)
        return [candidates[i][0] for i in indices]
    
    def _sample_fallback_triplets(self, K: int) -> List[Tuple[str, str, str]]:
        """后备采样：当无法满足邻近原则时使用"""
        all_nodes = list(self.nodes.keys())
        triplets = []
        attempts = 0
        max_attempts = K * 100
        
        while len(triplets) < K and attempts < max_attempts:
            attempts += 1
            selected = self.rng.choice(all_nodes, size=3, replace=False)
            triplet = tuple(selected)
            
            if not self._is_triplet_compared(triplet):
                triplets.append(triplet)
        
        return triplets
    
    def propose_triplet(self, K: int = 10) -> List[Dict[str, Any]]:
        """
        根据规则提出K个triplet用于比较。
        
        规则优先级:
        1. 有重复不抽原则
        2. 单节点温饱原则
        3. 邻近抽样原则
        4. 高分多抽原则
        
        Args:
            K: 需要的triplet数量
            
        Returns:
            K个triplet，每个元素包含3个节点的信息
        """
        if len(self.nodes) < 3:
            raise ValueError(f"节点数量不足: {len(self.nodes)} < 3")
        
        # 检查是否还有未温饱的节点
        hungry_nodes = self._get_hungry_nodes()
        
        if hungry_nodes:
            # 温饱阶段：优先让未温饱节点参与
            triplets = self._sample_hungry_triplets(K)
        else:
            # 邻近 + 高分多抽阶段
            triplets = self._sample_nearby_triplets(K)
        
        # 如果不足K个，用后备采样补充
        if len(triplets) < K:
            remaining = K - len(triplets)
            fallback = self._sample_fallback_triplets(remaining)
            
            # 过滤掉重复的
            existing = set(frozenset(t) for t in triplets)
            for t in fallback:
                if frozenset(t) not in existing:
                    triplets.append(t)
                    existing.add(frozenset(t))
                    if len(triplets) >= K:
                        break
        
        # 构造返回结果
        results = []
        for triplet in triplets[:K]:
            nodes_info = []
            for name in triplet:
                info = self.nodes[name].copy()
                info['name'] = name
                nodes_info.append(info)
            results.append({
                'triplet': list(triplet),
                'nodes': nodes_info
            })
        
        return results
    
    def submit_triplet_result(
        self, 
        results: List[Dict[str, Any]], 
        triplet_nodes: List[Dict[str, Any]]
    ) -> None:
        """
        提交triplet比较结果。
        
        Args:
            results: 比较结果列表，每个元素包含:
                - largest: 1/2/3 表示triplet中第几个是最大的
                - smallest: 1/2/3 表示triplet中第几个是最小的
            triplet_nodes: 对应的triplet节点信息（来自propose_triplet的返回）
        """
        if len(results) != len(triplet_nodes):
            raise ValueError(f"结果数量({len(results)})与triplet数量({len(triplet_nodes)})不匹配")
        
        for result, triplet_info in zip(results, triplet_nodes):
            triplet = triplet_info['triplet']
            largest_idx = result['largest'] - 1  # 转换为0-based
            smallest_idx = result['smallest'] - 1
            
            # 找出中间的
            middle_idx = 3 - largest_idx - smallest_idx
            
            largest_name = triplet[largest_idx]
            middle_name = triplet[middle_idx]
            smallest_name = triplet[smallest_idx]
            
            # 转换为pairwise比较结果（用于TrueSkill）
            # largest > middle, largest > smallest, middle > smallest
            comparisons = [
                {'img_name_A': largest_name, 'img_name_B': middle_name, 'compare_result': True},
                {'img_name_A': largest_name, 'img_name_B': smallest_name, 'compare_result': True},
                {'img_name_A': middle_name, 'img_name_B': smallest_name, 'compare_result': True},
            ]
            
            # 更新TrueSkill分数
            self.calculator.process_comparisons(comparisons)
            
            # 记录比较日志
            log = {
                'triplet': triplet,
                'largest': largest_name,
                'smallest': smallest_name,
                'middle': middle_name
            }
            self.comparison_logs.append(log)
            
            # 更新已比较集合
            self.compared_triplets.add(frozenset(triplet))
            
            # 更新参与次数
            for name in triplet:
                self.node_compare_counts[name] = self.node_compare_counts.get(name, 0) + 1
    
    def get_rankings(self) -> List[Tuple[str, float, float]]:
        """
        获取当前排名。
        
        Returns:
            排序后的列表，每个元素为 (name, mu, sigma)
        """
        final_scores = self.calculator.get_final_scores()
        
        # 添加未参与比较的节点
        default_mu = self.config['trueskill']['mu']
        default_sigma = self.config['trueskill']['sigma']
        
        for name in self.nodes:
            if name not in final_scores:
                final_scores[name] = {'mu': default_mu, 'sigma': default_sigma}
        
        # 按mu排序
        sorted_items = sorted(
            final_scores.items(),
            key=lambda x: x[1]['mu'],
            reverse=True
        )
        
        return [(name, data['mu'], data['sigma']) for name, data in sorted_items]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        hungry_nodes = self._get_hungry_nodes()
        
        return {
            'total_nodes': len(self.nodes),
            'total_comparisons': len(self.comparison_logs),
            'hungry_nodes': len(hungry_nodes),
            'avg_compare_per_node': np.mean(list(self.node_compare_counts.values())) if self.node_compare_counts else 0
        }
    
    def get_anchors(self) -> List[Dict[str, Any]]:
        """
        获取各个分位数（10%, 20%, ... 90%）附近的锚点样本。
        
        对于每个分位数 10i%，取正负2%区间内的数据，
        只考虑标注次数最多的5个样本，以 1/sigma^2 为权重进行抽样。
        
        Returns:
            list of dict，每个元素包含节点信息和 percentile 字段（0-100，分数越高分位数越高）
        """
        # 获取所有节点的分数信息
        final_scores = self.calculator.get_final_scores()
        
        if not final_scores:
            return []
        
        # 计算每个节点的分位数（基于mu排序，分数越高分位数越高）
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1]['mu'])
        n = len(sorted_items)
        
        node_percentiles = {}
        for rank, (name, scores) in enumerate(sorted_items):
            # 分位数：rank越高分数越高，所以 percentile = rank / (n-1) * 100
            percentile = (rank / max(n - 1, 1)) * 100
            node_percentiles[name] = {
                'percentile': percentile,
                'mu': scores['mu'],
                'sigma': scores['sigma'],
                'compare_count': self.node_compare_counts.get(name, 0)
            }
        
        anchors = []
        
        # 对于每个目标分位数（10%, 20%, ..., 90%）
        for target_pct in range(10, 100, 10):
            # 找出在 [target_pct - 2, target_pct + 2] 区间内的节点
            candidates = []
            for name, info in node_percentiles.items():
                if abs(info['percentile'] - target_pct) <= 2:
                    candidates.append((name, info))
            
            if not candidates:
                # 如果没有节点在区间内，放宽区间到 ±5
                for name, info in node_percentiles.items():
                    if abs(info['percentile'] - target_pct) <= 5:
                        candidates.append((name, info))
            
            if not candidates:
                continue
            
            # 只考虑标注次数最多的5个
            candidates.sort(key=lambda x: x[1]['compare_count'], reverse=True)
            top_candidates = candidates[:5]
            
            # 以 1/sigma^2 为权重进行抽样
            weights = np.array([1.0 / (c[1]['sigma'] ** 2) for c in top_candidates])
            weights = weights / weights.sum()
            
            # 加权抽样选择1个
            idx = self.rng.choice(len(top_candidates), p=weights)
            selected_name, selected_info = top_candidates[idx]
            
            # 构造返回结果
            anchor = self.nodes[selected_name].copy()
            anchor['name'] = selected_name
            anchor['percentile'] = selected_info['percentile']
            anchor['mu'] = selected_info['mu']
            anchor['sigma'] = selected_info['sigma']
            anchor['compare_count'] = selected_info['compare_count']
            anchors.append(anchor)
        
        return anchors
    
    def add_node_with_score(
        self, 
        node_info: Dict[str, Any], 
        relative_score: float
    ) -> str:
        """
        添加带有分数估计的节点。
        
        根据 relative_score（0-100）估计节点的初始 mu 值。
        
        Args:
            node_info: 节点信息
            relative_score: 相对分数（0-100），表示该节点在所有节点中的预估位置
            
        Returns:
            添加的节点名称
        """
        node_info = node_info.copy()
        
        # 获取或生成name
        if 'name' in node_info:
            name = node_info['name']
        else:
            name = self._generate_hash_name(node_info)
            node_info['name'] = name
        
        # 检查是否重复
        if name in self.nodes:
            print(f"警告: 节点 {name} 已存在，跳过")
            return name
        
        # 根据 relative_score 估计 mu
        # 需要根据现有节点的分数分布来计算
        final_scores = self.calculator.get_final_scores()
        
        if final_scores:
            # 按mu排序获取分布
            sorted_scores = sorted([s['mu'] for s in final_scores.values()])
            n = len(sorted_scores)
            
            # 根据 relative_score 插值计算 mu
            # relative_score=0 对应最低分，relative_score=100 对应最高分
            if n == 1:
                estimated_mu = sorted_scores[0]
            else:
                # 线性插值
                position = (relative_score / 100) * (n - 1)
                lower_idx = int(position)
                upper_idx = min(lower_idx + 1, n - 1)
                frac = position - lower_idx
                
                estimated_mu = sorted_scores[lower_idx] * (1 - frac) + sorted_scores[upper_idx] * frac
        else:
            # 没有现有分数，使用默认值
            default_mu = self.config['trueskill']['mu']
            # relative_score 偏移默认值
            estimated_mu = default_mu + (relative_score - 50) * 0.5
        
        # 添加节点
        self.nodes[name] = node_info
        self.node_compare_counts[name] = 0
        
        # 设置初始 TrueSkill rating（使用估计的 mu，但 sigma 仍然较大表示不确定）
        # 使用比默认稍小的 sigma，因为我们有一定的先验估计
        initial_sigma = self.config['trueskill']['sigma'] * 0.8
        self.calculator.ratings[name] = self.calculator.env.Rating(
            mu=estimated_mu, 
            sigma=initial_sigma
        )
        
        return name


if __name__ == "__main__":
    # 简单测试
    manager = EloManager()
    
    # 添加测试节点
    test_nodes = [
        {'name': 'node_1', 'value': 90},
        {'name': 'node_2', 'value': 70},
        {'name': 'node_3', 'value': 50},
        {'name': 'node_4', 'value': 30},
        {'name': 'node_5', 'value': 10},
    ]
    manager.add_nodes(test_nodes)
    
    # 提出triplet
    triplets = manager.propose_triplet(K=2)
    print(f"\n提出的triplet:")
    for t in triplets:
        print(f"  {t['triplet']}")
    
    # 模拟提交结果
    results = []
    for t in triplets:
        # 简单模拟：按value排序
        nodes = t['nodes']
        values = [n['value'] for n in nodes]
        largest = values.index(max(values)) + 1
        smallest = values.index(min(values)) + 1
        results.append({'largest': largest, 'smallest': smallest})
    
    manager.submit_triplet_result(results, triplets)
    
    # 查看排名
    print("\n当前排名:")
    for name, mu, sigma in manager.get_rankings():
        print(f"  {name}: mu={mu:.2f}, sigma={sigma:.2f}")
    
    print("\n统计信息:")
    print(manager.get_stats())

