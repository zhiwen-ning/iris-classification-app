鸢尾花分类预测工具
一个简单的机器学习项目，通过网页输入花萼 / 花瓣特征，预测鸢尾花品种（setosa、versicolor、virginica）。
功能简介
输入 4 个特征（花萼长度、宽度；花瓣长度、宽度）
实时返回预测结果（含品种名称和标签）
支持本地和 Docker 部署
快速使用（本地）
克隆代码，进入项目目录
激活虚拟环境：
bash
# Windows
.venv\Scripts\Activate.ps1
# Linux/macOS
source .venv/bin/activate
安装依赖：pip install -r requirements.txt
训练模型（首次运行）：python ml/train.py
启动后端：python app/main.py
打开 test.html 或通过 python -m http.server 8000 访问 http://localhost:8000/test.html
部署
测试环境：按上述步骤用虚拟环境启动
生产环境：用 Docker 构建镜像并启动（见 DEPLOYMENT.md）
示例数据
setosa：5.1, 3.5, 1.4, 0.2
versicolor：6.0, 2.8, 4.5, 1.5
virginica：6.5, 3.0, 5.5, 2.0