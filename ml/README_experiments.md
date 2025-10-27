MLflow 配置（实验跟踪）
默认存储：本地 mlruns/ 目录（自动生成）
启动 UI（查看实验）：mlflow ui，访问 http://localhost:5000
实验记录内容
每次训练记录：算法类型、超参数（如树深度）、准确率、模型文件路径
模型保存：最优模型自动保存到 mlruns/<run_id>/artifacts/model
模型选择流程
执行训练：python ml/train.py（自动记录到 MLflow）
通过 MLflow UI 查看各实验准确率
选择准确率最高的模型，在 app/main.py 中配置对应 run_id