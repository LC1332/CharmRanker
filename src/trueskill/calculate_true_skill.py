import trueskill
import json
import os
import re

class TrueSkillCalculator:
    """
    一个用于从 JSONL 文件加载比较数据并计算 TrueSkill 分数的类。
    """
    
    def __init__(self, mu=25.0, sigma=8.333, beta=4.166, tau=0.083):
        """
        初始化 TrueSkill 计算器。
        
        Args:
            mu (float): 初始分数的平均值。
            sigma (float): 初始分数的不确定度。
        """
        self.env = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau)
        self.ratings = {}
        
    def _get_or_create_rating(self, img_name):
        """获取或创建一张图片的 TrueSkill Rating 对象。"""
        if img_name not in self.ratings:
            self.ratings[img_name] = self.env.Rating()
        return self.ratings[img_name]

    def process_comparisons(self, comparisons):
        """
        处理一个包含比较结果的列表，并更新 TrueSkill 分数。
        
        Args:
            comparisons (list): 包含比较结果的字典列表。
        """
        for comp in comparisons:
            img_a = comp['img_name_A']
            img_b = comp['img_name_B']
            result = comp['compare_result']

            rating_a = self._get_or_create_rating(img_a)
            rating_b = self._get_or_create_rating(img_b)

            if result is True:
                # A > B, A is the winner
                new_rating_a, new_rating_b = self.env.rate([[rating_a], [rating_b]])
            else:
                # B > A, B is the winner
                new_rating_b, new_rating_a = self.env.rate([[rating_b], [rating_a]])
            
            self.ratings[img_a] = new_rating_a[0]
            self.ratings[img_b] = new_rating_b[0]
            
    def get_final_scores(self):
        """
        返回所有图片的最终 TrueSkill 分数。
        
        Returns:
            dict: 一个字典，键为图片名，值为包含 'mu' 和 'sigma' 的字典。
        """
        final_scores = {}
        for img_name, rating in self.ratings.items():
            final_scores[img_name] = {
                'mu': rating.mu,
                'sigma': rating.sigma
            }
        return final_scores

    @staticmethod
    def load_comparisons_from_jsonl(file_path):
        """
        静态方法：从 .jsonl 文件中读取比较结果数据。
        
        Args:
            file_path (str): .jsonl 文件的路径。
            
        Returns:
            list: 包含比较结果的字典列表。
        """
        comparisons = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    comparisons.append(json.loads(line.strip()))
            return comparisons
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Could not decode JSON from '{file_path}'. Details: {e}")
            return []
        
    def sort_final_score_by_mu(self, final_scores):
        sorted_scores = sorted(final_scores.items(), key=lambda item: item[1]['mu'], reverse=True)
        
        # 打印排序后的结果
        print("Calculated TrueSkill Scores (Sorted by mu):")
        for img_name, score_data in sorted_scores:
            print(f"  图片: {img_name}, 分数 (mu): {score_data['mu']:.2f}, 不确定度 (sigma): {score_data['sigma']:.2f}")

        return sorted_scores

    def sort_final_score_by_mu_and_sigma(self, final_scores):
        sorted_scores = sorted(final_scores.items(), key=lambda item: item[1]['mu'] - 3 * item[1]['sigma'], reverse=True)

        # 打印排序后的结果
        print("Calculated TrueSkill Scores (Sorted by mu - 3*sigma):")
        for img_name, score_data in sorted_scores:
            lower_bound = score_data['mu'] - 3 * score_data['sigma']
            print(f"  图片: {img_name}, 分数 (mu): {score_data['mu']:.2f}, 可信下限: {lower_bound:.2f}")

        return sorted_scores
        

def main():
    # 实例化 TrueSkill 计算器
    calculator = TrueSkillCalculator()
    
    # 加载比较数据
    comparisons_filepath = os.path.join(os.path.dirname(__file__), 'compare.jsonl')
    comparisons = calculator.load_comparisons_from_jsonl(comparisons_filepath)
    # 计算 TrueSkill 分数
    calculator.process_comparisons(comparisons)
    # 输出最终的 TrueSkill 分数
    final_scores = calculator.get_final_scores()
    print(json.dumps(calculator.sort_final_score_by_mu(final_scores), indent=4))
    print(json.dumps(calculator.sort_final_score_by_mu_and_sigma(final_scores), indent=4))
    # print(json.dumps(final_scores, indent=4))


if __name__ == '__main__':
    main()