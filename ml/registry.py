# ml/registry.py（完整修复版）
import mlflow.sklearn
import yaml
import os


def get_production_model():
    """加载生产模型（优先读取新训练的路径，自动验证有效性）"""
    model_path = ""
    current_model_file = "ml/registry/current_model.md"

    # 1. 优先读取新训练生成的current_model.md（重新训练后会自动更新）
    if os.path.exists(current_model_file):
        try:
            with open(current_model_file, "r", encoding="utf-8") as f:
                for line in f:
                    if "模型路径：" in line:
                        model_path = line.split("：")[-1].strip()
                        print(f"从current_model.md读取到模型路径：{model_path}")
                        break
            # 验证读取到的路径是否存在
            if model_path and os.path.exists(model_path):
                return mlflow.sklearn.load_model(model_path)
            else:
                print(
                    f"current_model.md中记录的路径无效：{model_path}，尝试查找最新模型"
                )
        except Exception as e:
            print(f"读取current_model.md出错：{str(e)}，尝试查找最新模型")

    # 2. 自动查找mlruns中最新的模型（避免依赖硬编码路径）
    mlruns_root = "mlruns"
    if os.path.exists(mlruns_root):
        # 递归查找所有包含MLmodel文件的模型目录（MLflow模型标志）
        model_dirs = []
        for root, dirs, files in os.walk(mlruns_root):
            if "MLmodel" in files:
                # 记录路径和最后修改时间（用于排序）
                model_dirs.append((root, os.path.getmtime(root)))
        # 按修改时间排序，取最新的模型
        if model_dirs:
            latest_model_path = sorted(model_dirs, key=lambda x: x[1], reverse=True)[0][
                0
            ]
            print(f"自动找到最新模型路径：{latest_model_path}")
            return mlflow.sklearn.load_model(latest_model_path)

    # 3. 所有方法失败，提示手动配置
    raise FileNotFoundError(
        f"""
    未找到有效模型！请按以下步骤操作：
    1. 确认已重新训练模型：python ml/train.py
    2. 训练完成后，手动复制新模型路径（从mlflow ui获取）：
       示例：mlruns/0/f4d46c1ca0c044f8ad51ad5f429c8138/artifacts/improved-model
    3. 打开 ml/registry/current_model.md，将"模型路径："后的内容替换为新路径
    """
    )
