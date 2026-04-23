# 场景素材最佳匹配引擎 

这是一个完全独立的、开箱即用的图像最佳匹配算法包。它封装了先进的不对称局部特征匹配算法与 DINOv2 / EfficientNet 视觉提取引擎，能够极其精准地从大量素材库中寻找到与场景图透视、角度或局部特征最吻合的素材。

## 特性
- **零依赖配置**：只需复制本文件夹，安装标准的 Python 科学计算库即可。
- **防分辨率丢失**：采用滑动窗口特征切片，支持 4K 及以上超高分辨率图片的匹配。
- **抗严重遮挡**：基于 Asymmetric Partial Matching，即使场景中的物体被严重遮挡（只露出一部分），也能准确找出素材库中的完整对应素材。
- **智能缓存隔离**：自动在当前目录下建立 `models` 缓存目录，避免了使用 PyTorch 默认的 C 盘 `.cache`，杜绝了跨系统运行时的权限与路径问题。

## 安装要求

请确保你的环境（或虚拟环境）中安装了 `requirements.txt` 中指定的依赖：

```bash
pip install -r requirements.txt
```

## 极简使用示例

只需几行代码，即可实现工业级的图片匹配：

```python
from scene_matcher import SceneMatcherAPI

# 1. 初始化匹配器
# 首次运行会自动下载模型权重，并缓存于本包目录下的 models 文件夹中
matcher = SceneMatcherAPI(model_type="dinov2_vitb14")

# 2. 传入两组图片路径，一键获取匹配映射！
# 假设你想给每一张场景图找到最匹配的前 2 张素材 (top_k=2)
scene_files = ["/path/to/scene_1.jpg", "/path/to/scene_2.jpg"]
material_files = ["/path/to/mat_A.png", "/path/to/mat_B.png", "/path/to/mat_C.png"]

results = matcher.match_images(scene_files, material_files, top_k=2)

# 打印结果查看
for scene, matched_materials in results.items():
    print(f"场景图: {scene}")
    for i, mat in enumerate(matched_materials):
        print(f"  -> 第 {i+1} 匹配: {mat}")
```

### 快捷文件夹处理方式

如果你不想手动遍历图片，也可以直接传入文件夹：

```python
from scene_matcher import SceneMatcherAPI

matcher = SceneMatcherAPI()

# 自动读取文件夹内的所有图片进行匹配
results = matcher.match_folders(
    scene_dir="/data/input/scenes",
    material_dir="/data/input/materials_white",
    top_k=1
)
```

## 注意事项
算法会自动检测机器是否含有兼容 CUDA 的显卡（GPU）。如果有，将自动启用硬件加速及 FP16 半精度推理；如果没有，会自动回退至纯 CPU 计算模式，无需任何手动干预。
