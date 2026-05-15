class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        min_point = init
        for one_iter in range(iterations): 
            min_point = min_point - learning_rate * 2 * min_point
        return round(min_point, 5)