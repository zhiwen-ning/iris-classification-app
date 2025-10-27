import os
import sys
import json
import pytest

# 核心：按项目根目录名称（iris-classification-app）定位，确保导入正确
current_script_path = os.path.abspath(__file__)
project_root = current_script_path
# 向上遍历，直到找到名为"iris-classification-app"的目录（项目根目录）
while "iris-classification-app" not in os.path.basename(project_root):
    project_root = os.path.dirname(project_root)
    # 防止遍历到系统根目录仍未找到（避免无限循环）
    if project_root == os.path.dirname(project_root):
        raise FileNotFoundError(
            "未找到项目根目录 'iris-classification-app'，请确认目录名称正确"
        )
# 将项目根目录添加到Python模块搜索路径
sys.path.insert(0, project_root)

# 现在可正常导入app模块
from app.main import app


@pytest.fixture
def client():
    """创建Flask测试客户端，模拟API请求"""
    app.config["TESTING"] = True  # 开启测试模式，关闭调试和异常捕获
    with app.test_client() as client:
        yield client


def test_api_normal_response(client):
    """测试API接收完整、合法参数时的正常响应"""
    # 构造符合要求的请求数据（鸢尾花setosa品种的典型特征）
    request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    # 发送POST请求到/predict接口
    response = client.post(
        "/predict",
        data=json.dumps(request_data),
        content_type="application/json"  # 明确指定JSON格式
    )

    # 验证响应结果
    assert response.status_code == 200, f"预期状态码200，实际为{response.status_code}"
    result = json.loads(response.data)
    assert result["status"] == "success", f"预期状态'success'，实际为{result['status']}"
    assert "predicted_species" in result, "响应中缺少'predicted_species'字段"
    assert "label" in result, "响应中缺少'label'字段（数字标签）"
    # 验证预测结果合理性（应为setosa相关标签，具体值根据编码映射调整）
    assert result["predicted_species"].lower() in ["iris-setosa", "setosa"], \
        f"预期预测为setosa，实际为{result['predicted_species']}"


def test_api_missing_param(client):
    """测试API接收缺失参数时的错误处理"""
    # 构造缺失参数的请求数据（缺少petal_width）
    request_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
        # 故意遗漏petal_width
    }
    # 发送POST请求
    response = client.post(
        "/predict",
        data=json.dumps(request_data),
        content_type="application/json"
    )

    # 验证错误响应
    assert response.status_code == 400, f"预期状态码400（参数错误），实际为{response.status_code}"
    result = json.loads(response.data)
    assert result["status"] == "fail", f"预期状态'fail'，实际为{result['status']}"
    assert "缺少参数" in result["error"] or "petal_width" in result["error"], \
        f"错误信息未提示缺少参数，实际为{result['error']}"