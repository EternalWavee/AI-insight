# 使用教程

本教程介绍如何在本地环境中配置并启动本项目服务。

---

## 1. 环境配置

推荐使用 **Conda + Python 3.11**。

### 1.1 创建并激活环境

```bash
conda create -n ai_insight python=3.11
conda activate ai_insight
```

## 2. 依赖安装
请确认当前位于项目根目录，执行：
```bash
pip install -r requirements.txt
```
如遇到网络问题，可使用国内镜像（可选）：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
## 3. 启动服务

在已激活的 conda 环境中运行：
```bash
python server.py
```

终端中将显示类似信息：
```bash
Running on http://127.0.0.1:5000
```
## 4. 使用方式

在浏览器中打开上述地址
进入 Web 页面后，可上传文件或输入文本
系统将自动执行 ASR / 文本处理 / 关键词提取流程
处理结果将在页面或接口返回中展示

## 5. 常见问题
### Q1：启动时报错找不到某个包？

请确认：

已正确激活 conda 环境

已执行 `pip install -r requirements.txt`

### Q2：ASR / Whisper 相关功能不可用？

请检查对应模块是否已正确安装
若网络受限，建议使用**科学上网**工具后再运行（因为模型是从huggingface下载的）