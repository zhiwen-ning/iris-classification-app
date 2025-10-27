# ml/data_pipeline.py（完整修复版）
import pandas as pd
import os
import yaml  # 新增：导入yaml模块
from sklearn.preprocessing import LabelEncoder  # 用于标签编码


def load_and_clean_data():
    raw_path = "data/raw/iris_v1.csv"
    processed_path = "data/processed/iris_v2.csv"
    os.makedirs("data/processed", exist_ok=True)

    # 检查原始数据是否存在
    if not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"原始数据文件不存在：{raw_path}\n请在data/raw/目录下放置iris_v1.csv"
        )

    # 读取原始数据（无表头，手动指定列名）
    column_names = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]
    df_raw = pd.read_csv(raw_path, names=column_names)

    # 修复SettingWithCopyWarning：用copy()创建独立DataFrame
    df_clean = df_raw.drop_duplicates().copy()

    # 标签编码（字符串→数字）
    le = LabelEncoder()
    df_clean["species"] = le.fit_transform(df_clean["species"].str.lower())
    print(f"数据清洗完成：{len(df_raw)}行 → {len(df_clean)}行（去重+标签编码）")

    # 保存带表头的清洗后数据
    df_clean.to_csv(processed_path, index=False, header=True)
    print(f"清洗后数据已保存至：{processed_path}")

    # 保存标签映射（已导入yaml，无NameError）
    os.makedirs("ml/registry", exist_ok=True)
    label_map = dict(zip(le.classes_, range(len(le.classes_))))
    with open("ml/registry/label_map.yml", "w", encoding="utf-8") as f:
        yaml.dump(label_map, f)
    print(f"标签映射已保存至：ml/registry/label_map.yml")

    return df_clean


if __name__ == "__main__":
    load_and_clean_data()
