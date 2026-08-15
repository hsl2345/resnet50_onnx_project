# ResNet50 图像分类推理优化服务

> 基于 FastAPI 搭建 ResNet50 图像分类服务,完成从原生 PyTorch 推理到 ONNX Runtime 的全流程优化落地,支持单张 / 批量图片推理,附带完整的对等性能基准测试体系,可直接作为生产级图像分类服务的基础模板。

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.19.2-purple.svg)](https://onnxruntime.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

---

## 目录

- [项目简介](#项目简介)
- [核心特性](#核心特性)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [环境要求](#环境要求)
- [快速开始](#快速开始)
  - [1. 克隆项目](#1-克隆项目)
  - [2. 创建虚拟环境并安装依赖](#2-创建虚拟环境并安装依赖)
  - [3. 导出 ONNX 模型](#3-导出-onnx-模型)
  - [4. 启动服务](#4-启动服务)
  - [5. 接口调用示例](#5-接口调用示例)
- [性能测试](#性能测试)
- [优化方案说明](#优化方案说明)
- [精度验证](#精度验证)
- [进阶配置](#进阶配置)
  - [切换推理引擎](#切换推理引擎)
  - [调整 ONNX 性能参数](#调整-onnx-性能参数)
  - [GPU 加速支持](#gpu-加速支持)
- [常见问题](#常见问题)
- [扩展方向](#扩展方向)
- [许可证](#许可证)
- [致谢](#致谢)

---

## 项目简介

本项目是 AI 模型部署入门级实战项目,基于 FastAPI 搭建 ResNet50 图像分类服务,完成了从原生 PyTorch 推理到 ONNX Runtime 全流程优化落地,支持单张 / 批量图片推理,附带完整的对等性能基准测试体系,可直接作为生产级图像分类服务的基础模板。

通过学习本项目可掌握:

- PyTorch 动态图模型到 ONNX 静态图的转换流程
- ONNX Runtime 图优化、线程对齐、内存池等推理加速手段
- FastAPI 异步图像推理服务的工程化封装
- 公平的双引擎性能基准测试方法(预热、固定输入、线程对齐)

---

## 核心特性

- **双推理引擎兼容**:同时支持原生 PyTorch 与 ONNX Runtime 推理引擎,对外接口完全一致,可一键切换
- **RESTful 规范接口**:提供单张 / 批量图片上传推理接口,配套 Swagger 可视化文档,开箱即用
- **全链路推理优化**:通过静态图转换、算子融合、常量折叠、线程对齐等手段,吞吐量提升 158%+
- **专业性能测试体系**:内置公平基准测试脚本,量化对比延迟、P95 尾部延迟、吞吐量、内存占用多维度指标
- **工程化项目结构**:遵循 Python 工业级项目规范,模块化封装,支持 Docker 容器化一键部署
- **动态 Batch 支持**:ONNX 模型支持动态批量大小,适配不同并发场景的推理需求
- **精度无损优化**:严格对齐预处理与后处理逻辑,优化后推理结果与原生模型误差 < 1e-5

---

## 技术栈

| 分类 | 技术选型 |
| --- | --- |
| Web 框架 | FastAPI + Uvicorn |
| 模型与推理 | PyTorch、TorchVision、ONNX、ONNX Runtime |
| 图像处理 | Pillow |
| 性能分析 | timeit、psutil、cProfile |
| 工程化 | Python 面向对象封装、配置分离、异常统一处理 |
| 部署支持 | Docker / Docker Compose(可扩展) |

---

## 项目结构

```
resnet_onnx_project/
├── app/                        # 核心业务代码
│   ├── __init__.py
│   ├── models/                 # 推理模型封装
│   │   ├── __init__.py
│   │   ├── torch_model.py      # 原生 PyTorch 推理类
│   │   └── onnx_model.py       # ONNX Runtime 优化推理类
│   ├── api/                    # 接口路由
│   │   ├── __init__.py
│   │   └── inference.py        # 推理接口实现
│   └── utils/                  # 工具函数
│       └── __init__.py
├── models/                     # 模型文件目录
│   └── resnet50_dynamic.onnx   # 导出的动态 Batch ONNX 模型
├── tests/                      # 测试与性能脚本
│   ├── export_onnx.py          # ONNX 模型导出脚本
│   ├── precision_verify.py     # 精度对齐验证脚本
│   └── benchmark.py            # 性能对比基准测试脚本
├── docs/                       # 文档与测试报告
│   └── performance_report.md   # 性能测试详细报告
├── main.py                     # FastAPI 服务入口
├── requirements.txt            # 项目依赖清单
├── .env                        # 环境配置文件
├── .gitignore                  # Git 忽略规则
└── README.md                   # 项目说明文档
```

> 说明:`test.jpg`、`__pycache__/`、`venv/` 等运行时产物已被 `.gitignore` 排除。

---

## 环境要求

### 硬件要求

| 组件 | 最低要求 | 推荐配置 |
| --- | --- | --- |
| CPU | 2 核 | 4 核及以上 |
| 内存 | 2GB | 4GB 及以上 |
| GPU | 可选 | NVIDIA GPU(可选,GPU 加速) |

### 软件要求

| 软件 | 版本要求 |
| --- | --- |
| 操作系统 | Linux / WSL2 / macOS(Windows 也可运行,推荐 WSL2) |
| Python | 3.9 ~ 3.11(推荐 3.10) |
| PyTorch | 2.4.0+ |
| ONNX Runtime | 1.19.2+ |
| Docker | 可选(用于容器化部署) |

---

## 快速开始

### 1. 克隆项目

```bash
git clone git@github.com:hsl2345/resnet50_onnx_project.git
cd resnet_onnx_project
```

### 2. 创建虚拟环境并安装依赖

```bash
# 创建虚拟环境
python -m venv venv

# Linux/WSL 激活虚拟环境
source venv/bin/activate

# Windows PowerShell 激活
# .\venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 导出 ONNX 模型

> 由于 ONNX 模型文件体积较大(约 98MB),未纳入版本控制。克隆后需自行导出,生成到 `models/resnet50_dynamic.onnx`。

```bash
python tests/export_onnx.py
```

### 4. 启动服务

```bash
# 开发模式启动
python main.py

# 生产模式启动
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
```

服务启动后访问:`http://localhost:8000/docs` 即可打开 Swagger 可视化接口文档,在线调试接口。

### 5. 接口调用示例

#### 健康检查

```bash
curl http://localhost:8000/health
```

#### 单张图片推理

```bash
curl -X POST "http://localhost:8000/inference/single" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test.jpg"
```

#### 批量图片推理

```bash
curl -X POST "http://localhost:8000/inference/batch" \
  -F "files=@test1.jpg" \
  -F "files=@test2.jpg"
```

---

## 性能测试

### 测试环境

| 环境项 | 参数 |
| --- | --- |
| 操作系统 | WSL2 Ubuntu |
| CPU | 4 核心 |
| Python 版本 | 3.10 |
| PyTorch 版本 | 2.4.0+cu121 |
| ONNX Runtime 版本 | 1.19.2 |
| 测试模型 | ResNet50 ImageNet 预训练 |
| 测试输入 | 1x3x224x224 单图推理 |

### 测试方案

- 模型仅全局加载一次,消除初始化开销对测试结果的影响
- 预热 10 轮,排除冷启动、算子编译的一次性开销
- 正式测试 200 轮,统计平均延迟、P95 尾部延迟、吞吐量、内存峰值
- 双引擎对齐 CPU 线程数,保证测试条件完全对等

### 测试结果

| 指标 | 原生 PyTorch | ONNX Runtime | 优化效果 |
| --- | --- | --- | --- |
| 单次平均延迟 | 47.76 ms | 18.46 ms | 降低 61.3% |
| P95 尾部延迟 | 71.82 ms | 18.90 ms | 降低 73.7%(3.8 倍优化) |
| 稳态吞吐量 | 20.94 FPS | 54.18 FPS | 提升 158.7%(2.59 倍优化) |
| 启动加载内存 | 119.87 MB | 178.47 MB | ONNX 略高(算子缓存预分配) |
| 运行峰值内存 | 604.76 MB | 703.16 MB | 高并发场景 ONNX 内存波动更稳定 |

### 复现测试

```bash
python tests/benchmark.py
```

详细测试报告见 [docs/performance_report.md](./docs/performance_report.md)。

---

## 优化方案说明

1. **静态图转换**:将 PyTorch 动态图模型导出为 ONNX 静态图格式,消除动态图的调度开销
2. **全量图优化**:开启 ONNX Runtime 全量图优化等级,实现常量折叠、算子融合、死代码消除
3. **并行性能优化**:对齐 CPU 物理核心数配置线程数,解决多线程过载导致的性能劣化问题
4. **内存优化**:配置内存池策略,平衡推理速度与内存占用
5. **工程化优化**:单例模式全局加载模型,复用输入张量内存,减少重复分配开销

---

## 精度验证

严格对齐 PyTorch 与 ONNX 版本的预处理、后处理逻辑,随机输入测试下:

- 平均绝对误差(MAE)< 1e-5
- 最大绝对误差 < 1e-4
- 分类结果完全一致,实现精度无损优化

验证命令:

```bash
python tests/precision_verify.py
```

---

## 进阶配置

### 切换推理引擎

修改 `app/api/inference.py` 中的导入语句,即可切换 PyTorch / ONNX Runtime 引擎,接口完全兼容:

```python
# 使用 PyTorch 原生引擎
from app.models.torch_model import TorchResNetClassifier
classifier = TorchResNetClassifier()

# 使用 ONNX Runtime 优化引擎
from app.models.onnx_model import ONNXResNetClassifier
classifier = ONNXResNetClassifier()
```

### 调整 ONNX 性能参数

修改 `app/models/onnx_model.py` 中的会话配置:

- `intra_op_num_threads`:算子内并行线程数,默认取 `os.cpu_count()`,也可通过环境变量 `ONNX_NUM_THREADS` 覆盖,建议设置为 CPU 物理核心数
- `enable_cpu_mem_arena`:内存池开关,关闭可降低内存占用,速度损失 < 5%
- `cpu_mem_limit`:CPU 内存池上限,适配内存受限环境

### GPU 加速支持

安装 GPU 版 ONNX Runtime 即可自动启用 CUDA 加速:

```bash
pip uninstall onnxruntime -y
pip install onnxruntime-gpu==1.19.2
```

---

## 常见问题

### Q1:启动时报 `ModuleNotFoundError: No module named 'app'`

**原因**:Python 找不到项目根目录下的 `app` 包。

**解决**:

- 确保在项目根目录 `resnet_onnx_project/` 下执行启动命令
- 或显式设置 `PYTHONPATH`: `export PYTHONPATH=$(pwd)`

### Q2:ONNX 推理比 PyTorch 还慢

**原因**:线程数设置不合理,线程竞争导致性能下降。

**解决**:

- 默认情况下 `app/models/onnx_model.py` 会自动取 `os.cpu_count()` 作为线程数,多数环境无需手动调整
- 如需自定义,设置环境变量 `ONNX_NUM_THREADS` 为 CPU 物理核心数(查看命令 `nproc`)
- 跑基准测试时,`tests/benchmark.py` 已通过 `OMP_NUM_THREADS` 等环境变量统一对齐 PyTorch 与 ONNX 的线程数

### Q3:导出 ONNX 时报 `opset_version` 不支持

**原因**:PyTorch 版本过低,不支持指定算子集版本。

**解决**:升级 PyTorch 至 2.0 以上,或将 `tests/export_onnx.py` 中 `opset_version=17` 调低至 13/14。

### Q4:接口上传图片后报 `请上传图片格式文件`

**原因**:`file.content_type` 未正确识别为 `image/*`。

**解决**:

- 确认文件确实是图片格式(jpg/png/webp 等)
- 部分客户端会省略 `Content-Type`,可改用 Postman 或 Swagger 文档上传

### Q5:精度验证误差过大

**原因**:导出 ONNX 时未切换 `model.eval()`,或预处理逻辑不一致。

**解决**:

- 确认 `tests/export_onnx.py` 中已调用 `model.eval()`
- 确认 ONNX 推理类使用的预处理参数与 PyTorch 版本完全一致

### Q6:批量推理接口返回 `没有有效的图片文件`

**原因**:上传的所有文件 `content_type` 都不以 `image/` 开头,被全部跳过。

**解决**:确认 `files` 字段使用复数形式 `-F "files=@xxx.jpg"`,且每个文件均为合法图片。

---

## 扩展方向

1. **容器化部署**:编写 Dockerfile 与 docker-compose.yml,实现服务一键容器化部署
2. **更高性能优化**:接入 TensorRT 执行提供程序,GPU 场景下进一步提升推理速度
3. **多模型支持**:扩展 YOLOv8、SAM2 等视觉模型,统一推理服务框架
4. **生产级特性**:新增限流、鉴权、日志收集、监控告警等功能
5. **并发压测**:接入 JMeter/hey 工具,测试服务高并发下的 QPS 与稳定性
6. **Triton 部署**:迁移至 NVIDIA Triton Inference Server,支持动态批处理、多模型调度

---

## 许可证

本项目代码采用 [MIT 许可证](./LICENSE) 开源。

ResNet50 模型权重遵循其自身的开源协议,详见 [PyTorch Vision Models](https://github.com/pytorch/vision)。

---

## 致谢

- [FastAPI](https://github.com/tiangolo/fastapi) - 高性能 Python Web 框架
- [PyTorch](https://github.com/pytorch/pytorch) - 深度学习框架
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) - 跨平台高性能推理引擎
- [TorchVision](https://github.com/pytorch/vision) - 提供预训练 ResNet50 权重
