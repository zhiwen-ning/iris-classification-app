鸢尾花分类项目 - 部署检查清单（极简版）
一、预发布环境（测试用）
步骤	操作	验证
1	激活 venv + pip install -r requirements.txt	终端显 (.venv)，无安装报错
2	有模型则看 mlruns/ 目录，无则跑 python ml/train.py	生成 mlruns/ 文件夹
3	开两个终端：
1. 后端：python app/main.py
2. 前端：python -m http.server 8000	后端显 http://127.0.0.1:5000，前端无报错
4	浏览器开 http://localhost:8000/test.html，输特征点预测	弹成功窗，显 “预测品种：xxx”
二、生产环境（Docker 用）
步骤	操作	验证
1	装 Docker + 克隆代码	跑 docker --version 显版本
2	requirements.txt 无拼写错
3	构建镜像：docker build -t iris-app:prod .	跑 docker images 能找到该镜像
4	启动容器：
docker run -d -p 5000:5000 -p 80:8000 --name iris-prod --restart always iris-app:prod	跑 docker ps 显容器 “Up”
5	浏览器开 http://服务器IP/test.html 测预测	网页能打开，预测结果对
三、常见问题解决
端口占了：查进程 netstat -ano | findstr :端口号，杀进程 taskkill /PID 号 /F
跨域错：确认 main.py 有 CORS(app)
模型加载错：跑 python ml/train.py 重新生成 mlruns/