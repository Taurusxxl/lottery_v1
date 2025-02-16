# Feature Engineering System / 特征工程系统
import tensorflow as tf
import numpy as np
import logging
from tensorflow.keras.layers import Conv1D, Dense, Lambda

# 获取logger实例
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """特征工程类"""
    
    def __init__(self):
        """初始化特征工程组件"""
        self.logger = logging.getLogger(__name__)
    
    def build_all_features(self, x):
        """构建所有特征"""
        try:
            # 1. 基础特征
            basic_features = self._build_basic_features(x)
            
            # 2. 高级特征
            advanced_features = self._build_advanced_features(x)
            
            # 3. 数字特征
            digit_features = self._build_advanced_digit_features(x)
            
            # 4. 形态特征
            pattern_features = self._build_pattern_features(x)
            
            # 5. 特征融合
            all_features = tf.keras.layers.Concatenate()([
                basic_features,
                advanced_features,
                digit_features,
                pattern_features
            ])
            
            return all_features
            
        except Exception as e:
            self.logger.error(f"构建特征时出错: {str(e)}")
            return x

    def _build_advanced_features(self, x):
        """构建高级特征分析"""
        features = []
        
        # 1. 冷热号分析
        hot_cold = self._analyze_hot_cold_numbers(x)
        features.append(hot_cold)
        
        # 2. 号码频率统计
        freq = self._analyze_frequency(x)
        features.append(freq)
        
        # 3. 和值分析
        sum_features = self._analyze_sum_value(x)
        features.append(sum_features)
        
        # 4. 数字特征分析
        digit_features = self._analyze_digit_patterns(x)
        features.append(digit_features)
        
        # 5. 形态分析
        pattern_features = self._analyze_number_patterns(x)
        features.append(pattern_features)
        
        # 6. 012路分析
        route_features = self._analyze_012_routes(x)
        features.append(route_features)
        
        return tf.keras.layers.Concatenate()(features)

    def _analyze_hot_cold_numbers(self, x, window_sizes=[100, 500, 1000]):
        """分析冷热号"""
        features = []
        
        for window in window_sizes:
            # 最近window期的频率
            recent = x[:, -window:]
            freq = tf.reduce_sum(tf.one_hot(tf.cast(recent, tf.int32), 10), axis=1)
            features.append(freq)
        
        return tf.keras.layers.Concatenate()(features)

    def _analyze_digit_patterns(self, x):
        """分析数字特征"""
        # 1. 奇偶比
        odd_even = tf.reduce_sum(tf.cast(x % 2 == 1, tf.float32), axis=-1, keepdims=True)
        
        # 2. 大小比 (5-9为大)
        big_small = tf.reduce_sum(tf.cast(x >= 5, tf.float32), axis=-1, keepdims=True)
        
        # 3. 质合比
        prime_numbers = tf.constant([2, 3, 5, 7])
        is_prime = tf.reduce_sum(tf.cast(
            tf.equal(x[..., None], prime_numbers), tf.float32
        ), axis=-1)
        prime_composite = tf.reduce_sum(is_prime, axis=-1, keepdims=True)
        
        # 4. 跨度
        span = tf.reduce_max(x, axis=-1) - tf.reduce_min(x, axis=-1)
        
        return tf.concat([odd_even, big_small, prime_composite, span[..., None]], axis=-1)

    def _analyze_sum_value(self, x):
        """分析和值特征"""
        # 1. 计算和值
        sum_value = tf.reduce_sum(x, axis=-1, keepdims=True)
        
        # 2. 和值分布区间
        sum_ranges = [
            (0, 10), (11, 20), (21, 30), (31, 40), (41, 45)
        ]
        sum_dist = []
        for low, high in sum_ranges:
            in_range = tf.logical_and(
                sum_value >= low,
                sum_value <= high
            )
            sum_dist.append(tf.cast(in_range, tf.float32))
        
        # 3. 和值特征
        sum_features = tf.concat([sum_value, tf.concat(sum_dist, axis=-1)], axis=-1)
        
        return sum_features

    def _analyze_012_routes(self, x):
        """分析012路特征"""
        # 1. 计算每个数字的路数
        routes = tf.math.floormod(x, 3)  # 对3取余
        
        # 2. 统计每个位置的路数分布
        route_distributions = []
        for i in range(5):  # 五个位置
            digit_routes = routes[..., i:i+1]
            # 统计0,1,2路的数量
            route_counts = []
            for r in range(3):
                count = tf.reduce_sum(
                    tf.cast(digit_routes == r, tf.float32),
                    axis=-1, keepdims=True
                )
                route_counts.append(count)
            route_distributions.append(tf.concat(route_counts, axis=-1))
        
        # 3. 计算整体012路比例
        total_route_dist = tf.reduce_mean(tf.concat(route_distributions, axis=-1), axis=-1, keepdims=True)
        
        # 4. 分析路数组合特征
        route_patterns = self._analyze_route_patterns(routes)
        
        # 5. 计算相邻位置的路数关系
        route_transitions = []
        for i in range(4):
            transition = tf.cast(
                routes[..., i:i+1] == routes[..., i+1:i+2],
                tf.float32
            )
            route_transitions.append(transition)
        
        # 6. 特征组合
        features = [
            *route_distributions,  # 每位路数分布
            total_route_dist,     # 整体路数比例
            route_patterns,       # 路数组合特征
            *route_transitions    # 相邻位置路数关系
        ]
        
        return tf.concat(features, axis=-1)

    def _analyze_route_patterns(self, routes):
        """分析路数组合模式"""
        # 1. 全0路
        all_zero = tf.reduce_all(routes == 0, axis=-1, keepdims=True)
        
        # 2. 全1路
        all_one = tf.reduce_all(routes == 1, axis=-1, keepdims=True)
        
        # 3. 全2路
        all_two = tf.reduce_all(routes == 2, axis=-1, keepdims=True)
        
        # 4. 012路是否均匀分布(各有至少一个)
        has_zero = tf.reduce_any(routes == 0, axis=-1, keepdims=True)
        has_one = tf.reduce_any(routes == 1, axis=-1, keepdims=True)
        has_two = tf.reduce_any(routes == 2, axis=-1, keepdims=True)
        balanced = tf.logical_and(
            tf.logical_and(has_zero, has_one),
            has_two
        )
        
        # 5. 主路特征(出现最多的路数)
        route_counts = []
        for r in range(3):
            count = tf.reduce_sum(
                tf.cast(routes == r, tf.float32),
                axis=-1, keepdims=True
            )
            route_counts.append(count)
        main_route = tf.argmax(tf.concat(route_counts, axis=-1), axis=-1)
        
        return tf.concat([
            tf.cast(all_zero, tf.float32),
            tf.cast(all_one, tf.float32),
            tf.cast(all_two, tf.float32),
            tf.cast(balanced, tf.float32),
            tf.cast(main_route, tf.float32)[..., tf.newaxis]
        ], axis=-1)

    def _build_advanced_digit_features(self, x):
        """构建高级数字特征"""
        features = []
        
        # 1. 每位数字的独立特征
        for i in range(5):
            digit = x[..., i:i+1]  # 提取第i位数字
            
            # 数字频率统计
            freq = tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.histogram_fixed_width(x, [0, 9], nbins=10), tf.float32)
            )(digit)
            
            # 数字转换模式
            transitions = self._build_digit_transitions(digit)
            
            features.extend([freq, transitions])
        
        # 2. 数字组合特征
        for i in range(5):
            for j in range(i+1, 5):
                # 两位数字组合
                pair = tf.stack([x[..., i], x[..., j]], axis=-1)
                pair_features = self._build_pair_features(pair)
                features.append(pair_features)
        
        # 3. 完整号码特征
        full_number = tf.reshape(x, (-1, x.shape[1]))  # 将5位数字合并为一个完整号码
        number_features = self._build_number_features(full_number)
        features.append(number_features)
        
        return tf.keras.layers.Concatenate()(features)

    def _build_digit_transitions(self, digit):
        """构建数字转换特征"""
        # 计算相邻数字之间的转换
        transitions = digit[:, 1:] - digit[:, :-1]
        # 转换为one-hot编码
        transitions_one_hot = tf.one_hot(tf.cast(transitions + 9, tf.int32), 19)  # -9到9共19种可能
        return tf.reduce_mean(transitions_one_hot, axis=1)

    def _build_pair_features(self, pair):
        """构建数字对特征"""
        # 计算数字对的差值
        diff = tf.abs(pair[..., 0] - pair[..., 1])
        # 计算数字对的和
        sum_pair = pair[..., 0] + pair[..., 1]
        # 计算数字对的乘积
        prod = pair[..., 0] * pair[..., 1]
        
        return tf.stack([diff, sum_pair, prod], axis=-1)

    def _build_number_features(self, numbers):
        """构建完整号码特征"""
        # 1. 计算整体统计特征
        mean = tf.reduce_mean(numbers, axis=-1, keepdims=True)
        std = tf.math.reduce_std(numbers, axis=-1, keepdims=True)
        
        # 2. 计算号码的数字频率分布
        freq_dist = tf.keras.layers.Lambda(
            lambda x: tf.cast(tf.histogram_fixed_width(x, [0, 9], nbins=10), tf.float32)
        )(numbers)
        
        return tf.concat([mean, std, freq_dist], axis=-1)

    def _analyze_frequency(self, x):
        """分析号码频率"""
        # 转换为整数类型
        x = tf.cast(x, tf.int32)
        
        # 计算每个数字的出现频率
        freq = tf.zeros_like(x, dtype=tf.float32)
        for i in range(10):
            mask = tf.cast(x == i, tf.float32)
            freq += mask * tf.reduce_mean(mask, axis=1, keepdims=True)
        
        return freq

    def _analyze_number_patterns(self, x):
        """分析号码形态"""
        # 1. 连号分析
        consecutive = tf.reduce_sum(tf.cast(
            x[:, 1:] == x[:, :-1] + 1, tf.float32
        ), axis=-1, keepdims=True)
        
        # 2. 重复号分析
        unique_counts = tf.reduce_sum(tf.one_hot(tf.cast(x, tf.int32), 10), axis=-2)
        repeats = tf.reduce_sum(tf.cast(unique_counts > 1, tf.float32), axis=-1, keepdims=True)
        
        # 3. 形态识别
        patterns = self._identify_patterns(x)
        
        return tf.concat([consecutive, repeats, patterns], axis=-1)

    def _identify_patterns(self, x):
        """识别号码形态"""
        patterns = []
        
        # 1. 豹子号(AAAAA)
        baozi = tf.reduce_all(x == x[..., :1], axis=-1, keepdims=True)
        patterns.append(tf.cast(baozi, tf.float32))
        
        # 2. 组5(AAAAB)
        sorted_x = tf.sort(x, axis=-1)
        zu5 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :4] == sorted_x[..., :1], tf.float32), axis=-1) == 4,
            sorted_x[..., 4] != sorted_x[..., 0]
        )
        patterns.append(tf.cast(zu5, tf.float32)[..., tf.newaxis])
        
        # 3. 组10(AAABB)
        zu10 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :3] == sorted_x[..., :1], tf.float32), axis=-1) == 3,
            tf.reduce_sum(tf.cast(sorted_x[..., 3:] == sorted_x[..., 3:4], tf.float32), axis=-1) == 2
        )
        patterns.append(tf.cast(zu10, tf.float32)[..., tf.newaxis])
        
        # 4. 组20(AAABC)
        zu20 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :3] == sorted_x[..., :1], tf.float32), axis=-1) == 3,
            sorted_x[..., 3] != sorted_x[..., 4]
        )
        patterns.append(tf.cast(zu20, tf.float32)[..., tf.newaxis])
        
        # 5. 组30(AABBC)
        zu30 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :2] == sorted_x[..., :1], tf.float32), axis=-1) == 2,
            tf.reduce_sum(tf.cast(sorted_x[..., 2:4] == sorted_x[..., 2:3], tf.float32), axis=-1) == 2
        )
        patterns.append(tf.cast(zu30, tf.float32)[..., tf.newaxis])
        
        # 6. 组60(AABCD)
        zu60 = tf.logical_and(
            tf.reduce_sum(tf.cast(sorted_x[..., :2] == sorted_x[..., :1], tf.float32), axis=-1) == 2,
            tf.reduce_all(sorted_x[..., 2:] != sorted_x[..., 1:4], axis=-1)
        )
        patterns.append(tf.cast(zu60, tf.float32)[..., tf.newaxis])
        
        # 7. 组120(ABCDE)
        zu120 = tf.reduce_all(sorted_x[..., 1:] > sorted_x[..., :-1], axis=-1, keepdims=True)
        patterns.append(tf.cast(zu120, tf.float32))
        
        return tf.concat(patterns, axis=-1)

    def _build_periodic_features(self, x, periods=[60, 120, 360, 720, 1440]):
        """构建周期性特征"""
        features = []
        
        for period in periods:
            # 1. 提取周期模式
            pattern = self._extract_periodic_pattern(x, period)
            
            # 2. 周期性偏差
            deviation = x - pattern
            
            # 3. 周期强度
            strength = tf.reduce_mean(tf.abs(pattern), axis=-1, keepdims=True)
            
            features.extend([pattern, deviation, strength])
            
        return tf.keras.layers.Concatenate()(features)

    def _extract_periodic_pattern(self, x, period):
        """提取周期性模式"""
        # 重塑以匹配周期
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]
        n_periods = length // period
        
        # 重塑为(batch, n_periods, period, features)
        x_reshaped = tf.reshape(x[:, :n_periods*period], 
                               (batch_size, n_periods, period, -1))
        
        # 计算周期内模式
        pattern = tf.reduce_mean(x_reshaped, axis=1)  # 平均周期模式
        variance = tf.math.reduce_variance(x_reshaped, axis=1)  # 周期变异性
        
        return tf.concat([pattern, variance], axis=-1)

    def _build_probability_features(self, x):
        """构建概率分布特征"""
        # 1. 条件概率矩阵
        cond_matrix = tf.zeros((10, 10, 5))  # 每位数字的转移概率
        
        # 2. 计算历史转移概率
        for i in range(5):
            current = tf.cast(x[..., i], tf.int32)
            next_digit = tf.roll(current, shift=-1, axis=0)
            
            # 更新转移矩阵
            for j in range(10):
                for k in range(10):
                    mask_current = tf.cast(current == j, tf.float32)
                    mask_next = tf.cast(next_digit == k, tf.float32)
                    prob = tf.reduce_mean(mask_current * mask_next)
                    cond_matrix = tf.tensor_scatter_nd_update(
                        cond_matrix,
                        [[j, k, i]],
                        [prob]
                    )
        
        return cond_matrix

    def _build_statistical_features(self, x):
        """构建统计特征"""
        # 1. 移动统计
        windows = [60, 360, 720]  # 1小时、6小时、12小时
        stats = []
        
        for window in windows:
            # 移动平均
            ma = tf.keras.layers.AveragePooling1D(
                pool_size=window, strides=1, padding='same')(x)
            # 移动标准差
            std = tf.math.reduce_std(
                tf.stack([x, ma], axis=-1), axis=-1)
            # 移动极差
            pooled_max = tf.keras.layers.MaxPooling1D(
                pool_size=window, strides=1, padding='same')(x)
            pooled_min = -tf.keras.layers.MaxPooling1D(
                pool_size=window, strides=1, padding='same')(-x)
            range_stat = pooled_max - pooled_min
            
            stats.extend([ma, std, range_stat])
        
        # 2. 概率分布特征
        probs = self._build_probability_features(x)
        
        return tf.concat(stats + [probs], axis=-1)

    def _build_trend_features(self, x):
        """构建趋势特征"""
        # 1. 短期趋势
        short_ma = tf.keras.layers.AveragePooling1D(
            pool_size=12, strides=1, padding='same')(x)
        short_trend = tf.sign(x - short_ma)
        
        # 2. 中期趋势
        medium_ma = tf.keras.layers.AveragePooling1D(
            pool_size=60, strides=1, padding='same')(x)
        medium_trend = tf.sign(x - medium_ma)
        
        # 3. 长期趋势
        long_ma = tf.keras.layers.AveragePooling1D(
            pool_size=360, strides=1, padding='same')(x)
        long_trend = tf.sign(x - long_ma)
        
        # 4. 趋势一致性
        trend_consistency = tf.reduce_mean(
            tf.cast(short_trend == medium_trend, tf.float32) * 
            tf.cast(medium_trend == long_trend, tf.float32),
            axis=-1, keepdims=True
        )
        
        return tf.concat([short_trend, medium_trend, long_trend, trend_consistency], axis=-1)

    def _build_correlation_features(self, x):
        """构建相关性特征"""
        # 1. 位置间相关性
        correlations = []
        for i in range(5):
            for j in range(i+1, 5):
                corr = self._compute_correlation(x[..., i], x[..., j])
                correlations.append(corr)
        
        # 2. 滞后相关性
        lag_correlations = []
        for lag in [1, 2, 3, 5, 10]:
            lagged_corr = self._compute_lag_correlation(x, lag)
            lag_correlations.append(lagged_corr)
        
        return tf.concat([*correlations, *lag_correlations], axis=-1)

    def _compute_lag_correlation(self, x, lag):
        """计算滞后相关性"""
        x_current = x[:, lag:]
        x_lagged = x[:, :-lag]
        
        return self._compute_correlation(x_current, x_lagged)

    def _build_pattern_features(self, x):
        """构建形态特征分析"""
        # 1. 当前形态识别
        current_patterns = self._identify_patterns(x)
        
        # 2. 形态遗漏值分析
        pattern_gaps = self._analyze_pattern_gaps(x)
        
        # 3. 形态转换规律
        pattern_transitions = self._analyze_pattern_transitions(x)
        
        # 4. 形态组合特征
        pattern_combinations = self._analyze_pattern_combinations(x)
        
        # 5. 形态周期性分析
        pattern_periodicity = self._analyze_pattern_periodicity(x)
        
        return tf.keras.layers.Concatenate()([
            current_patterns,
            pattern_gaps,
            pattern_transitions,
            pattern_combinations,
            pattern_periodicity
        ])

    def _analyze_pattern_combinations(self, x):
        """分析形态组合特征"""
        # 1. 计算所有可能的形态组合
        patterns = []
        for i in range(5):
            for j in range(i+1, 5):
                pair = tf.stack([x[..., i], x[..., j]], axis=-1)
                patterns.append(self._analyze_digit_pair(pair))
                
        # 2. 组合形态间的关联性
        pattern_corr = tf.stack([
            self._compute_pattern_correlation(p1, p2)
            for i, p1 in enumerate(patterns)
            for j, p2 in enumerate(patterns) if i < j
        ], axis=-1)
        
        return tf.concat([*patterns, pattern_corr], axis=-1)

    def _analyze_pattern_periodicity(self, x):
        """分析形态周期性"""
        periods = [12, 24, 60, 120, 360]
        periodicity = []
        
        for period in periods:
            # 1. 周期性模式提取
            pattern = self._extract_pattern_cycle(x, period)
            
            # 2. 周期强度计算
            strength = self._compute_cycle_strength(pattern)
            
            # 3. 周期稳定性分析
            stability = self._analyze_cycle_stability(pattern)
            
            periodicity.extend([pattern, strength, stability])
            
        return tf.concat(periodicity, axis=-1)

    def _extract_pattern_cycle(self, x, period):
        """提取形态周期模式"""
        # 1. 重塑数据以匹配周期
        batch_size = tf.shape(x)[0]
        n_cycles = tf.shape(x)[1] // period
        x_cycles = tf.reshape(x[:, :n_cycles*period], 
                            [batch_size, n_cycles, period, -1])
        
        # 2. 计算周期内的形态分布
        cycle_patterns = tf.reduce_mean(x_cycles, axis=1)
        
        # 3. 计算周期间的变异性
        cycle_variance = tf.math.reduce_variance(x_cycles, axis=1)
        
        return tf.concat([cycle_patterns, cycle_variance], axis=-1)

    def _compute_cycle_strength(self, pattern):
        """计算周期强度"""
        # 1. 自相关分析
        autocorr = tf.keras.layers.Conv1D(
            filters=1, kernel_size=pattern.shape[1],
            padding='same'
        )(pattern)
        
        # 2. 周期性强度评分
        strength = tf.reduce_mean(tf.abs(autocorr), axis=1, keepdims=True)
        
        return strength

    def _analyze_cycle_stability(self, pattern):
        """分析周期稳定性"""
        # 1. 计算相邻周期的差异
        diffs = pattern[:, 1:] - pattern[:, :-1]
        
        # 2. 计算稳定性指标
        stability = tf.reduce_mean(tf.abs(diffs), axis=1, keepdims=True)
        stability = tf.exp(-stability)  # 转换到0-1范围
        
        return stability

    def _analyze_digit_pair(self, pair):
        """分析数字对特征"""
        # 1. 计算数字对基本特征
        diff = tf.abs(pair[..., 0] - pair[..., 1])
        sum_pair = pair[..., 0] + pair[..., 1]
        prod = pair[..., 0] * pair[..., 1]
        
        # 2. 计算数字对的位置关系
        is_adjacent = tf.cast(diff == 1, tf.float32)
        is_complementary = tf.cast(sum_pair == 9, tf.float32)
        
        # 3. 计算组合特征
        features = tf.stack([
            diff, sum_pair, prod,
            is_adjacent, is_complementary
        ], axis=-1)
        
        return features

    def _compute_pattern_correlation(self, p1, p2):
        """计算形态相关性"""
        # 标准化
        p1_norm = (p1 - tf.reduce_mean(p1)) / (tf.math.reduce_std(p1) + 1e-6)
        p2_norm = (p2 - tf.reduce_mean(p2)) / (tf.math.reduce_std(p2) + 1e-6)
        
        # 计算相关系数
        corr = tf.reduce_mean(p1_norm * p2_norm, axis=-1, keepdims=True)
        
        return corr

    def _analyze_pattern_gaps(self, x):
        """分析形态遗漏值"""
        try:
            # 1. 获取形态
            patterns = self._identify_patterns(x)
            
            # 2. 初始化遗漏值计数器
            gap_counters = tf.zeros_like(patterns)
            
            # 3. 计算每种形态的遗漏值
            def update_gaps(sequence):
                gaps = []
                for i in range(7):  # 7种形态
                    last_pos = -1
                    current_gap = 0
                    pattern_positions = tf.where(sequence[:, i])
                    
                    if tf.size(pattern_positions) > 0:
                        last_pos = tf.reduce_max(pattern_positions)
                        current_gap = tf.shape(sequence)[0] - 1 - last_pos
                    
                    gaps.append(current_gap)
                return tf.stack(gaps)
            
            # 4. 应用遗漏值计算
            gaps = tf.keras.layers.Lambda(update_gaps)(patterns)
            
            # 5. 构建遗漏值特征
            gap_features = []
            
            # 当前遗漏值
            gap_features.append(gaps)
            
            # 历史最大遗漏值
            max_gaps = tf.reduce_max(gaps, axis=0, keepdims=True)
            gap_features.append(max_gaps)
            
            # 历史平均遗漏值
            mean_gaps = tf.reduce_mean(gaps, axis=0, keepdims=True)
            gap_features.append(mean_gaps)
            
            # 遗漏值分布
            gap_dist = tf.keras.layers.Lambda(
                lambda x: tf.cast(tf.histogram_fixed_width(x, [0, 1000], nbins=50), tf.float32)
            )(gaps)
            gap_features.append(gap_dist)
            
            return tf.concat(gap_features, axis=-1)
            
        except Exception as e:
            logger.error(f"分析形态遗漏值时出错: {str(e)}")
            return tf.zeros_like(x)

    def _analyze_pattern_transitions(self, x):
        """分析形态转换规律"""
        try:
            # 1. 识别所有形态
            patterns = self._identify_patterns(x)
            
            # 2. 计算形态转换矩阵
            def get_transition_matrix(sequence):
                matrix = tf.zeros((7, 7))  # 7x7转换矩阵
                for i in range(len(sequence)-1):
                    current = tf.argmax(sequence[i])
                    next_pattern = tf.argmax(sequence[i+1])
                    matrix = tf.tensor_scatter_nd_update(
                        matrix,
                        [[current, next_pattern]],
                        [1.0]
                    )
                return matrix
            
            transition_matrix = tf.keras.layers.Lambda(get_transition_matrix)(patterns)
            
            # 3. 提取转换特征
            transitions = []
            
            # 转换概率
            prob_matrix = transition_matrix / (tf.reduce_sum(transition_matrix, axis=-1, keepdims=True) + 1e-7)
            transitions.append(tf.reshape(prob_matrix, [-1]))
            
            # 最常见转换路径
            common_transitions = tf.reduce_max(prob_matrix, axis=-1)
            transitions.append(common_transitions)
            
            # 形态稳定性(自我转换概率)
            stability = tf.linalg.diag_part(prob_matrix)
            transitions.append(stability)
            
            return tf.concat(transitions, axis=-1)
            
        except Exception as e:
            logger.error(f"分析形态转换规律时出错: {str(e)}")
            return tf.zeros_like(x)

    def _build_basic_features(self, x):
        """构建基础特征"""
        try:
            # 1. 时间特征
            time_features = self._add_positional_encoding(x)
            
            # 2. 多尺度特征
            multi_scale = self._build_multi_scale_features(x)
            
            # 3. 统计特征
            statistical = self._build_statistical_features(x)
            
            # 合并所有基础特征
            x = tf.concat([time_features, multi_scale, statistical], axis=-1)
            return x
            
        except Exception as e:
            logger.error(f"构建基础特征时出错: {str(e)}")
            return x

    def _add_positional_encoding(self, x):
        """添加位置编码"""
        seq_len = tf.shape(x)[1]
        d_model = tf.shape(x)[-1]
        
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        
        pe = tf.zeros((seq_len, d_model))
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([tf.range(seq_len), tf.range(0, d_model, 2)], axis=1),
            tf.sin(position * div_term)
        )
        pe = tf.tensor_scatter_nd_update(
            pe,
            tf.stack([tf.range(seq_len), tf.range(1, d_model, 2)], axis=1),
            tf.cos(position * div_term)
        )
        
        return x + pe[tf.newaxis, :, :]

    def _build_model_features(self, x):
        """构建模型特征"""
        # 1. 基础特征
        base_features = self._build_basic_features(x)
        
        # 2. 时序特征
        temporal_features = self._build_temporal_features(x)
        
        # 3. 模式特征
        pattern_features = self._build_pattern_features(x)
        
        # 4. 高阶特征组合
        combined_features = self._build_combined_features([
            base_features,
            temporal_features, 
            pattern_features
        ])
        
        return combined_features

    def _build_combined_features(self, feature_list):
        """构建高阶特征组合"""
        try:
            # 1. 特征连接
            x = tf.keras.layers.Concatenate()(feature_list)
            
            # 2. 非线性变换
            x = tf.keras.layers.Dense(256, activation='relu')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Dropout(0.2)(x)
            
            # 3. 特征交互
            x = self._build_feature_interactions(x)
            
            return x
            
        except Exception as e:
            logger.error(f"构建组合特征时出错: {str(e)}")
            return tf.zeros_like(x)

    def _build_feature_interactions(self, x):
        """构建特征交互"""
        try:
            # 1. 自注意力交互
            att = tf.keras.layers.MultiHeadAttention(
                num_heads=4,
                key_dim=32
            )(x, x)
            x = tf.keras.layers.Add()([x, att])
            x = tf.keras.layers.LayerNormalization()(x)
            
            # 2. 非线性特征组合
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            
            return x
            
        except Exception as e:
            logger.error(f"构建特征交互时出错: {str(e)}")
            return tf.zeros_like(x)

    def _build_temporal_features(self, x):
        """构建时序特征"""
        try:
            # 1. 时间编码
            time_encoding = self._add_temporal_encoding(x)
            
            # 2. 周期特征
            periodic = self._build_periodic_features(x)
            
            # 3. 趋势特征
            trend = self._build_trend_features(x)
            
            # 4. 相关性特征
            correlation = self._build_correlation_features(x)
            
            return tf.concat([time_encoding, periodic, trend, correlation], axis=-1)
            
        except Exception as e:
            logger.error(f"构建时序特征时出错: {str(e)}")
            return x

    def _add_temporal_encoding(self, x):
        """添加时间编码"""
        try:
            seq_len = tf.shape(x)[1]
            d_model = tf.shape(x)[-1]
            
            # 1. 位置编码
            position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
            div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
            
            pe = tf.zeros((seq_len, d_model))
            pe = tf.tensor_scatter_nd_update(
                pe,
                tf.stack([tf.range(seq_len), tf.range(0, d_model, 2)], axis=1),
                tf.sin(position * div_term)
            )
            pe = tf.tensor_scatter_nd_update(
                pe,
                tf.stack([tf.range(seq_len), tf.range(1, d_model, 2)], axis=1),
                tf.cos(position * div_term)
            )
            
            # 2. 周期性时间特征
            day_in_week = tf.cast(tf.math.floormod(position, 7), tf.float32) / 7.0
            hour_in_day = tf.cast(tf.math.floormod(position, 24), tf.float32) / 24.0
            minute_in_hour = tf.cast(tf.math.floormod(position, 60), tf.float32) / 60.0
            
            time_features = tf.concat([
                pe,
                day_in_week,
                hour_in_day,
                minute_in_hour
            ], axis=-1)
            
            return time_features[tf.newaxis, :, :]
            
        except Exception as e:
            logger.error(f"添加时间编码时出错: {str(e)}")
            return x

    def evaluate_model_feature_importance(self, model, X):
        """评估模型特征重要性"""
        try:
            # 使用模型的权重评估特征重要性
            feature_weights = model.get_layer('feature_layer').get_weights()[0]
            importance = np.abs(feature_weights).mean(axis=1)
            return importance
            
        except Exception as e:
            logger.error(f"评估特征重要性时出错: {str(e)}")
            return None
    
    def select_top_features(self, importance, top_k=10):
        """选择最重要的特征"""
        try:
            top_indices = np.argsort(importance)[-top_k:]
            return top_indices
            
        except Exception as e:
            logger.error(f"选择重要特征时出错: {str(e)}")
            return None

# 创建全局实例
feature_engineering = FeatureEngineering()
