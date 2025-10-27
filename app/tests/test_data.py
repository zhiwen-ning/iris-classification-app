import os
import sys
import pandas as pd
import pytest

# 核心：按项目根目录名称定位，确保导入正确
current_script_path = os.path.abspath(__file__)
project_root = current_script_path
while "iris-classification-app" not in os.path.basename(project_root):
    project_root = os.path.dirname(project_root)
    if project_root == os.path.dirname(project_root):
        raise FileNotFoundError(
            "未找到项目根目录 'iris-classification-app'，请确认目录名称正确"
        )
sys.path.insert(0, project_root)

# 现在可正常导入ml模块
from ml.data_pipeline import load_and_clean_data


def test_feature_range():
    """测试清洗后的数据特征是否在鸢尾花的合理生物学范围内"""
    df = load_and_clean_data()
    # 鸢尾花特征的典型合理范围（参考UCI数据集标准）
    valid_ranges = {
        "sepal_length": (4.3, 7.9),  # 花萼长度：4.3-7.9cm
        "sepal_width": (2.0, 4.4),  # 花萼宽度：2.0-4.4cm
        "petal_length": (1.0, 6.9),  # 花瓣长度：1.0-6.9cm
        "petal_width": (0.1, 2.5),  # 花瓣宽度：0.1-2.5cm
    }

    # 逐个验证特征范围
    for feature, (min_val, max_val) in valid_ranges.items():
        actual_min = df[feature].min()
        actual_max = df[feature].max()
        assert (
            actual_min >= min_val
        ), f"{feature}最小值异常：实际{actual_min} < 合理最小值{min_val}"
        assert (
            actual_max <= max_val
        ), f"{feature}最大值异常：实际{actual_max} > 合理最大值{max_val}"


def test_no_duplicates():
    """测试清洗后的数据无重复行"""
    df = load_and_clean_data()
    # 去重前的行数（清洗函数已去重，此处验证去重效果）
    df_with_duplicates = df.copy()
    df_with_duplicates = pd.concat(
        [df_with_duplicates, df.iloc[0:1]], ignore_index=True
    )  # 手动加1行重复数据
    assert len(df) == len(
        df.drop_duplicates()
    ), f"数据存在重复行：去重前{len(df)}行，去重后{len(df.drop_duplicates())}行"


def test_no_missing_values():
    """测试清洗后的数据无缺失值"""
    df = load_and_clean_data()
    # 计算所有列的缺失值总数
    total_missing = df.isnull().sum().sum()
    assert (
        total_missing == 0
    ), f"数据存在缺失值：共{total_missing}个缺失值，各列缺失情况：\n{df.isnull().sum()}"


def test_label_categories():
    """测试标签列编码正确（仅包含0/1/2三个类别，且每个类别样本数充足）"""
    df = load_and_clean_data()
    # 验证标签仅包含0/1/2
    unique_labels = set(df["species"].unique())
    assert unique_labels == {
        0,
        1,
        2,
    }, f"标签编码异常：实际类别{unique_labels}，预期{0,1,2}"

    # 验证每个类别样本数不少于10个（避免极端不平衡）
    label_counts = df["species"].value_counts()
    for label, count in label_counts.items():
        assert count >= 10, f"类别{label}样本数不足：仅{count}个，需至少10个"
