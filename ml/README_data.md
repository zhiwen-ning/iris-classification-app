数据概况
核心数据：鸢尾花数据集（iris.csv），包含 3 个品种、150 条样本，每条含 4 个特征（花萼长度 / 宽度、花瓣长度 / 宽度）
存储路径：data/iris.csv
DVC 配置（数据版本管理）
初始化：项目根目录执行 dvc init
跟踪数据：dvc add data/iris.csv（生成 data/iris.csv.dvc，需提交 Git）
远程存储：默认本地远程 ../dvc_remote，配置命令：
bash
dvc remote add -d myremote ../dvc_remote
同步数据：
推送到远程：dvc push
拉取数据：dvc pull
数据版本控制
数据更新后：dvc add data/iris.csv → dvc push → 提交 .dvc 文件到 Git
查看历史版本：git log data/iris.csv.dvc