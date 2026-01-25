"""
测试 calculate_trueskill_scores 函数的正确性。
"""

import unittest
from calculate_trueskill_scores import calculate_trueskill_scores


class TestCalculateTrueskillScores(unittest.TestCase):
    """测试 TrueSkill 分数计算函数"""
    
    def test_simple_chain(self):
        """测试简单链式关系：A > B > C"""
        edges = [
            ("A", "B"),  # A 胜 B
            ("B", "C"),  # B 胜 C
        ]
        scores = calculate_trueskill_scores(edges, seed=42)
        
        # 验证分数顺序
        self.assertGreater(scores["A"], scores["B"])
        self.assertGreater(scores["B"], scores["C"])
    
    def test_complete_order(self):
        """测试完全序关系：A > B > C，包含所有比较"""
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("A", "C"),
        ]
        scores = calculate_trueskill_scores(edges, seed=42)
        
        # 验证分数顺序
        self.assertGreater(scores["A"], scores["B"])
        self.assertGreater(scores["B"], scores["C"])
        self.assertGreater(scores["A"], scores["C"])
    
    def test_empty_edges(self):
        """测试空边表"""
        edges = []
        scores = calculate_trueskill_scores(edges)
        
        self.assertEqual(scores, {})
    
    def test_single_comparison(self):
        """测试单次比较"""
        edges = [("winner", "loser")]
        scores = calculate_trueskill_scores(edges, seed=42)
        
        self.assertEqual(len(scores), 2)
        self.assertIn("winner", scores)
        self.assertIn("loser", scores)
        self.assertGreater(scores["winner"], scores["loser"])
    
    def test_reproducibility_with_seed(self):
        """测试使用相同 seed 结果可复现"""
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "D"),
            ("A", "D"),
        ]
        
        scores1 = calculate_trueskill_scores(edges, seed=123)
        scores2 = calculate_trueskill_scores(edges, seed=123)
        
        for node in scores1:
            self.assertAlmostEqual(scores1[node], scores2[node], places=10)
    
    def test_multiple_wins(self):
        """测试多次胜利累积效果"""
        edges = [
            ("champion", "player1"),
            ("champion", "player2"),
            ("champion", "player3"),
            ("player1", "player2"),
            ("player2", "player3"),
        ]
        scores = calculate_trueskill_scores(edges, seed=42)
        
        # champion 全胜，应该分数最高
        for player in ["player1", "player2", "player3"]:
            self.assertGreater(scores["champion"], scores[player])
    
    def test_circular_relation(self):
        """测试循环关系：A > B, B > C, C > A（石头剪刀布）"""
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("C", "A"),
        ]
        scores = calculate_trueskill_scores(edges, seed=42)
        
        # 验证所有节点都有分数
        self.assertEqual(len(scores), 3)
        self.assertIn("A", scores)
        self.assertIn("B", scores)
        self.assertIn("C", scores)
        
        # 循环关系下分数应该比较接近
        all_scores = list(scores.values())
        score_range = max(all_scores) - min(all_scores)
        self.assertLess(score_range, 10)  # 分数差异不应过大
    
    def test_string_node_names(self):
        """测试各种字符串节点名"""
        edges = [
            ("image_001.jpg", "image_002.jpg"),
            ("image_002.jpg", "photo with spaces.png"),
            ("image_001.jpg", "photo with spaces.png"),
        ]
        scores = calculate_trueskill_scores(edges, seed=42)
        
        self.assertEqual(len(scores), 3)
        self.assertGreater(scores["image_001.jpg"], scores["image_002.jpg"])
        self.assertGreater(scores["image_002.jpg"], scores["photo with spaces.png"])
    
    def test_shuffle_effect(self):
        """测试 shuffle 次数对结果的影响"""
        edges = [
            ("A", "B"),
            ("B", "C"),
            ("A", "C"),
            ("D", "E"),
            ("E", "F"),
        ]
        
        # 使用不同的 shuffle 次数
        scores_1 = calculate_trueskill_scores(edges, n_shuffles=1, seed=42)
        scores_5 = calculate_trueskill_scores(edges, n_shuffles=5, seed=42)
        scores_10 = calculate_trueskill_scores(edges, n_shuffles=10, seed=42)
        
        # 基本顺序关系应该保持
        for scores in [scores_1, scores_5, scores_10]:
            self.assertGreater(scores["A"], scores["B"])
            self.assertGreater(scores["B"], scores["C"])
    
    def test_output_type(self):
        """测试输出类型正确"""
        edges = [("A", "B")]
        scores = calculate_trueskill_scores(edges, seed=42)
        
        self.assertIsInstance(scores, dict)
        for key, value in scores.items():
            self.assertIsInstance(key, str)
            self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()

