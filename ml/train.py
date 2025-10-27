import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder  # 新增：导入LabelEncoder
import yaml
import os


def load_config(config_path="ml/configs/train_config.yml"):
    """加载超参数配置（修复编码错误）"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}\n请确认ml/configs/目录下有train_config.yml")
    try:
        with open(config_path, "r", encoding="utf-8", errors="ignore") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(f"读取配置文件失败：{str(e)}\n建议重新用UTF-8编码保存train_config.yml")
    
    required_keys = ["baseline", "improved"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"配置文件缺少必要字段：{key}\n请确保配置包含baseline和improved节点")
    
    return config


def load_data(data_path="data/processed/iris_v2.csv"):
    """加载数据（修复重复表头问题：文件已有表头，不再手动指定）"""
    # 1. 检查文件是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据集不存在：{data_path}\n请先运行ml/data_pipeline.py生成iris_v2.csv")
    
    # 2. 关键修复：文件已有表头（如sepal_length），直接读取，不指定names参数
    try:
        df = pd.read_csv(data_path)  # 移除 names=column_names，避免重复添加表头
    except Exception as e:
        raise RuntimeError(f"读取数据集失败：{str(e)}\n请确认iris_v2.csv格式为'带表头的CSV'")
    
    # 3. 检查关键列是否存在（确保文件表头正确）
    required_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"数据集缺少关键列：{missing_cols}\n请确认CSV表头包含{required_cols}")
    
    # 4. 处理标签（若为字符串则自动编码）
    if df["species"].dtype == "object":
        print("检测到字符串标签，自动进行编码...")
        le = LabelEncoder()
        df["species"] = le.fit_transform(df["species"].str.lower())
        # 保存标签映射
        os.makedirs("ml/registry", exist_ok=True)
        label_map = dict(zip(le.classes_, range(len(le.classes_))))
        with open("ml/registry/label_map.yml", "w", encoding="utf-8") as f:
            yaml.dump(label_map, f)
        print(f"自动编码完成：标签映射已保存至 ml/registry/label_map.yml")
    
    # 5. 验证特征列是否为数值类型（避免表头重复导致的字符串数据）
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"特征列'{col}'必须是数值类型，当前是{df[col].dtype}\n原因：CSV文件可能重复添加了表头")
    
    return df


if __name__ == "__main__":
    try:
        # 1. 加载配置
        config = load_config()
        
        # 2. 加载数据（自动处理字符串标签）
        df = load_data()
        X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
        y = df["species"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"数据加载完成：训练集{X_train.shape} | 测试集{X_test.shape}")

        # 3. 基准模型实验
        print("\n=== 运行基准模型实验 ===")
        with mlflow.start_run(run_name="Baseline Model"):
            mlflow.log_param("git_commit", "e4f5g6h")
            mlflow.log_param("dvc_data_hash", "abc123")
            baseline_lr = config["baseline"].get("lr", 0.1)
            baseline_max_iter = config["baseline"].get("max_iter", 100)
            mlflow.log_param("learning_rate", baseline_lr)
            mlflow.log_param("max_iter", baseline_max_iter)
            
            model = LogisticRegression(max_iter=baseline_max_iter)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            mlflow.log_metric("test_accuracy", accuracy)
            print(f"基准模型测试准确率：{accuracy:.4f}")
            mlflow.sklearn.log_model(model, "baseline_model")

        # 4. 优化模型实验
        print("\n=== 运行优化模型实验 ===")
        with mlflow.start_run(run_name="Improved Model"):
            mlflow.log_param("git_commit", "e4f5g6h")
            mlflow.log_param("dvc_data_hash", "abc123")
            improved_lr = config["improved"].get("lr", 0.01)
            improved_max_iter = config["improved"].get("max_iter", 200)
            mlflow.log_param("learning_rate", improved_lr)
            mlflow.log_param("max_iter", improved_max_iter)
            
            model = LogisticRegression(max_iter=improved_max_iter, C=0.5)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy_improved = accuracy_score(y_test, y_pred)
            mlflow.log_metric("test_accuracy", accuracy_improved)
            print(f"优化模型测试准确率：{accuracy_improved:.4f}")
            mlflow.sklearn.log_model(model, "improved_model")

        # 5. 注册最优模型
        print("\n=== 注册最优模型 ===")
        best_model_name = "Improved Model" if accuracy_improved > accuracy else "Baseline Model"
        best_model_artifact_path = "improved_model" if accuracy_improved > accuracy else "baseline_model"
        best_run_id = mlflow.last_active_run().info.run_id
        best_model_full_path = f"mlruns/0/{best_run_id}/artifacts/{best_model_artifact_path}"
        
        os.makedirs("ml/registry", exist_ok=True)
        with open("ml/registry/current_model.md", "w", encoding="utf-8") as f:
            f.write(f"# 生产环境模型记录\n")
            f.write(f"- 最优模型：{best_model_name}\n")
            f.write(f"- 测试准确率：{max(accuracy, accuracy_improved):.4f}\n")
            f.write(f"- 模型路径：{best_model_full_path}\n")
            f.write(f"- Git Commit：e4f5g6h\n")
            f.write(f"- DVC数据哈希：abc123\n")
        
        print(f"\n训练完成！最优模型已注册到：{best_model_full_path}")
        print(f"查看实验详情：执行 `mlflow ui` 后访问 http://127.0.0.1:5000")

    except Exception as e:
        print(f"\n训练过程出错：{str(e)}")
        raise