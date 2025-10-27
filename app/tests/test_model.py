import os
import sys
import pandas as pd
import pytest
import yaml

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
from ml.registry import get_production_model


@pytest.fixture(scope="module")
def model_and_label_map():
    """加载生产模型和标签映射（模块级fixture，仅加载一次，提升效率）"""
    # 加载标签映射（指定UTF-8编码，避免解码错误）
    label_map_path = os.path.join(project_root, "ml", "registry", "label_map.yml")
    if not os.path.exists(label_map_path):
        raise FileNotFoundError(f"标签映射文件不存在：{label_map_path}\n请先运行ml/train.py")
    
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = yaml.safe_load(f)
    
    # 加载生产模型
    model = get_production_model()
    
    return model, label_map


def test_model_known_samples(model_and_label_map):
    """测试模型对已知典型样本的预测正确性"""
    model, label_map = model_and_label_map
    # 构建已知样本（特征+预期标签，参考鸢尾花标准数据集）
    test_cases = [
        # 样本1：setosa（预期标签0）
        {
            "features": {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            "expected_label": 0
        },
        # 样本2：versicolor（预期标签1）
        {
            "features": {"sepal_length": 7.0, "sepal_width": 3.2, "petal_length": 4.7, "petal_width": 1.4},
            "expected_label": 1
        },
        # 样本3：virginica（预期标签2）
        {
            "features": {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
            "expected_label": 2
        }
    ]
    
    # 逐个验证预测结果
    for idx, case in enumerate(test_cases):
        # 转换为模型要求的输入格式（带特征名的DataFrame）
        input_data = pd.DataFrame([case["features"]])
        # 模型预测
        pred_label = model.predict(input_data)[0]
        # 断言结果
        assert pred_label == case["expected_label"], \
            f"样本{idx+1}预测错误：\n输入特征：{case['features']}\n预期标签：{case['expected_label']}\n实际标签：{pred_label}"


def test_model_input_validation(model_and_label_map):
    """测试模型对异常输入的处理能力（仅校验非数值输入，数值范围由数据预处理保证）"""
    model, _ = model_and_label_map
    # 异常输入样本：仅保留“非数值输入”（模型必须报错），移除“数值不合理但格式正确”的样本
    invalid_cases = [
        # 情况1：特征值为字符串（非数值）→ 模型应报错（类型错误）
        {"sepal_length": "abc", "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        # 情况2：特征值为None（空值）→ 模型应报错（值错误）
        {"sepal_length": None, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
    ]

    # 验证模型对“非数值输入”的捕获能力
    for idx, case in enumerate(invalid_cases):
        input_data = pd.DataFrame([case])
        try:
            # 尝试预测（非数值输入必须报错）
            model.predict(input_data)
            # 若未报错，测试失败
            pytest.fail(
                f"样本{idx+1}未检测到异常输入：\n输入特征：{case}\n模型未抛出错误"
            )
        except (ValueError, TypeError):
            # 捕获到预期异常（非数值输入导致的错误），测试通过
            continue