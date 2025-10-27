from flask import Flask, request, jsonify
import mlflow.sklearn
import pandas as pd
import os
import yaml
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)


# --------------------------
# 标签映射加载（保持容错）
# --------------------------
def load_label_map():
    label_map_path = os.path.join(
        os.path.dirname(__file__), "../ml/registry/label_map.yml"
    )
    if os.path.exists(label_map_path):
        try:
            with open(label_map_path, "r", encoding="utf-8") as f:
                label_map = yaml.safe_load(f)
            if label_map is None:
                print(f"警告：{label_map_path} 为空，使用默认映射")
                return {0: "setosa", 1: "versicolor", 2: "virginica"}
            reverse_map = {v: k.replace("iris-", "") for k, v in label_map.items()}
            print(f"标签映射：{reverse_map}")
            return reverse_map
        except Exception as e:
            print(f"加载标签映射失败：{e}，使用默认")
    print(f"未找到{label_map_path}，使用默认映射")
    return {0: "setosa", 1: "versicolor", 2: "virginica"}


SPECIES_MAP = load_label_map()


# --------------------------
# 核心修复：模型路径验证+自动重试
# --------------------------
def get_valid_model_path():
    """
    验证模型路径是否存在，若不存在则：
    1. 尝试从mlruns目录查找最新模型
    2. 若找不到则提示用户重新训练
    """
    # 1. 优先从current_model.md读取
    current_model_file = os.path.join(
        os.path.dirname(__file__), "../ml/registry/current_model.md"
    )
    candidate_paths = []
    if os.path.exists(current_model_file):
        try:
            with open(current_model_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "模型路径：" in line:
                        path = line.split("：")[-1].strip()
                        candidate_paths.append(path)
        except Exception as e:
            print(f"读取current_model.md失败：{e}")

    # 2. 从.env添加候选路径
    env_path = os.getenv("MODEL_PATH")
    if env_path:
        candidate_paths.append(env_path)

    # 3. 验证候选路径是否存在
    for path in candidate_paths:
        # 处理路径格式（兼容Windows反斜杠）
        normalized_path = os.path.normpath(path)
        if os.path.exists(normalized_path):
            print(f"找到有效模型路径：{normalized_path}")
            return normalized_path

    # 4. 自动查找mlruns目录下的最新模型（最后修改的模型）
    mlruns_root = os.path.join(os.path.dirname(__file__), "../mlruns")
    if os.path.exists(mlruns_root):
        print(f"在{mlruns_root}中查找最新模型...")
        # 递归查找所有模型目录（特征：包含MLmodel文件）
        model_dirs = []
        for root, dirs, files in os.walk(mlruns_root):
            if "MLmodel" in files:  # MLflow模型的标志文件
                model_dirs.append((root, os.path.getmtime(root)))  # (路径, 修改时间)
        # 按修改时间排序，取最新的
        if model_dirs:
            latest_model = sorted(model_dirs, key=lambda x: x[1], reverse=True)[0][0]
            print(f"自动找到最新模型：{latest_model}")
            return latest_model

    # 5. 所有方法都失败，提示用户重新训练
    raise FileNotFoundError(
        """
    未找到有效模型路径！请执行以下步骤：
    1. 运行 python ml/train.py 重新训练模型（确保无错误）
    2. 训练完成后，mlruns目录会生成新模型
    3. 若仍失败，手动删除mlruns目录后重试
    """
    )


# --------------------------
# 加载模型（确保路径有效）
# --------------------------
try:
    model_path = get_valid_model_path()
    model = mlflow.sklearn.load_model(model_path)
    print("✅ 模型加载成功")
except Exception as e:
    print(f"❌ 模型加载失败：{str(e)}")
    raise  # 启动失败，必须解决模型问题


# --------------------------
# API接口（保持不变）
# --------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"status": "fail", "error": "需为application/json"}), 400
        data = request.json

        required_params = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        missing_params = [p for p in required_params if p not in data]
        if missing_params:
            return (
                jsonify(
                    {
                        "status": "fail",
                        "error": f"缺少参数：{', '.join(missing_params)}",
                    }
                ),
                400,
            )

        try:
            features = {k: float(data[k]) for k in required_params}
        except ValueError:
            return jsonify({"status": "fail", "error": "参数需为数字"}), 400

        input_df = pd.DataFrame([features])
        pred_label = model.predict(input_df)[0]
        return (
            jsonify(
                {
                    "status": "success",
                    "predicted_species": SPECIES_MAP[pred_label],
                    "label": int(pred_label),
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"status": "fail", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
