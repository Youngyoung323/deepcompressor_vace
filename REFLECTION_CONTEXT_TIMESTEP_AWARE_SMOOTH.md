# DeepCompressor Timestep-Aware Smooth Reflection Context

## 任务结论

- 在 `deepcompressor_vace` 里，按 `timestep` 分开统计激活并分别计算 `smooth_scale`，从“统计计算”角度是可行的。
- 但如果要做**真正的 per-timestep smooth**，即推理时不同 timestep 使用不同 `smooth_scale` 且保持 smooth 的线性等价关系，那么当前架构下**不是一个小改动**。
- 根本原因是：DeepCompressor 的 smooth 不是只在运行时给激活乘/除一个 scale，而是会**静态改写权重**，因此一旦 `smooth_scale` 依赖 timestep，匹配的权重变换也必须依赖 timestep。

## 当前 smooth 的真实数据流

### 1. 校准缓存中其实有 timestep

- Wan 校准样本的 `.pt` 文件里明确保存了 `input_kwargs["timestep"]`。
- `WanCalibCacheLoader.iter_samples()` 也会把这个 `timestep` 送回模型前向。

这说明“样本级 timestep 信息存在”。

### 2. 但 smooth 统计时，激活被按 sample 维直接拼接

- `ConcatCacheAction.apply()` 会把所有样本的张量沿 dim0 直接拷进一个大 tensor。
- 这一步只保留激活值，不保留“这一行来自哪个 timestep”的并行元数据。

所以当前 cache 结构本质上是：

```python
all_samples_x = concat([x_sample_0, x_sample_1, ...], dim=0)
```

而不是：

```python
{
  timestep_0: [...],
  timestep_1: [...],
}
```

### 3. smooth 计算阶段会把样本进一步展平

- `TensorCache.get_standardized_data()` 会把 `channels_dim` 之前的维度全部 flatten。
- `get_smooth_span()` / `ChannelMetric.abs_max()` / `root_mean_square()` 都是基于这个展平后的 tensor 去做跨样本归约。

也就是说，当前 smooth 的 span 计算天然是“把所有 timestep 混在一起”的。

### 4. 当前 smooth 是静态重参数化，不是纯运行时缩放

- `smooth_linear_modules()` 会先求出一个 `scale`
- 然后对目标线性层权重执行 `smooth_upscale_param(module.weight, scale, channels_dim=1)`
- 如果可 fuse，还会对前一层权重/偏置执行 `smooth_downscale_param(...)`
- 如果不能 fuse，才额外注册 `ActivationSmoother` hook，在运行时对输入 tensor 做一次除法/乘法

因此它的数学形式是：

```python
y = W x
=> y = (W * S) (x / S)
```

或者在 fuse 到前层时，把 `1/S` 吸收到前层参数里。

## 为什么“每个 timestep 一个 smooth_scale”不是小改动

## 关键约束

- 如果某一层在 timestep `t1` 用的是 `S_t1`
- 在 timestep `t2` 用的是 `S_t2`
- 那么与之匹配、保持等价的权重也必须分别变成：

```python
W_t1 = W * S_t1
W_t2 = W * S_t2
```

或者等价地，对上一层做不同的 `1 / S_t` 融合。

所以问题不只是“运行时根据 timestep 选一个 scale”：

- 还必须让**权重侧同步按 timestep 切换**
- 否则就不再是 smooth 的等价重参数化，只是单边改了激活

## 当前实现中的三个具体障碍

### 障碍 1：运行时 hook 拿不到 timestep 来选 scale

- `ActivationSmoother.process()` 的签名只有 `process(tensor)`
- `ProcessHook` 也是把 unpack 出来的每个 tensor 单独送进 `processor.process(x)`
- processor 本身拿不到完整 `input_kwargs`

所以现有 hook 机制无法根据 `timestep` 动态选不同 scale。

### 障碍 2：校准 cache 不保留 sample -> timestep 映射

- `ConcatCacheAction` 只拼接 tensor，不保存标签
- `smooth` 统计阶段看不到“这一段激活属于哪个 timestep”

所以即使要先离线算 `scale[timestep]`，当前 cache 结构也不够。

### 障碍 3：搜索评估只保留一份 `eval_kwargs`

- `WanCalibCacheLoader.iter_layer_activations()` 最后只拿 `layer_inputs[0].kwargs` 生成一份 `layer_kwargs`
- `SearchBasedCalibrator` 在评估多个 cached samples 时，会反复调用 `ipts.extract(i, eval_kwargs)`，这里的 `eval_kwargs` 是同一份

这意味着当前 search-based smooth 校准默认假设“所有 cached sample 共享同一组 eval kwargs”，并不支持每个 sample 各自的 timestep。

## 可行性判断

## 1. 只做“按 timestep 分开统计 span”

这是可行的。

做法可以是：

- 在 cache 阶段额外保存每个 sample 的 timestep 标签
- 将 `x_tensors` 按 timestep 分桶
- 对每个 timestep 分别算 `x_span[t]`
- 再得到 `scale[t]`

这一步只是统计逻辑扩展。

## 2. 做“真正的 per-timestep smooth”

理论上可行，但当前架构下代价很高。

至少需要新增下面几类能力：

- `cache` 侧保留 timestep 标签，或直接按 timestep 建桶
- `SmoothCalibrator` 支持输出 `dict[timestep, scale]`
- 运行时 hook 能读取当前 `timestep` 并选择对应 scale
- 权重侧也必须能按 timestep 动态切换

其中最后一项最难，因为当前权重是静态改写的。

## 可能的实现方向

### 方向 A：每个 timestep 存一份平滑后的权重

形式上最直接：

```python
W_t = W * S_t
```

运行时根据 timestep 选对应 `W_t` 和 `S_t`。

问题：

- 显存/存储开销极大
- 对 Wan 这种大模型不现实

### 方向 B：运行时动态重算平滑权重

即保留原权重 `W`，每次前向按当前 timestep 现算 `W * S_t`。

问题：

- 推理时每层每步都要重算或重量化权重
- 代价非常高

### 方向 C：只对激活做 timestep-aware scale，权重保持静态

这个改动最小，但**不再是严格的 smooth quant**

因为它对应的是：

```python
y = W (x / S_t)
```

而不是：

```python
y = (W * S_t) (x / S_t)
```

所以会直接改变函数本身，风险很大。

### 方向 D：折中成 timestep-bin / stage-aware smooth

这是最值得考虑的工程折中方案。

例如把 timestep 分成少数几个区间：

- early
- middle
- late

然后每个区间一套 scale，甚至一套权重副本。

这样：

- 比 per-timestep 精细
- 比 50/1000 个 timestep 的完整动态方案便宜很多

但依然需要支持“运行时按阶段切换 scale / 权重”。

## 对当前 DeepCompressor/Wan 的实际建议

- 如果目标是“先验证 timestep 信息是否值得利用”，建议先做**timestep-aware span analysis**，不要一开始就做真正动态 smooth。
- 第一阶段更合理的实验是：
  - 为每层统计 `x_span[timestep, channel]`
  - 观察不同 timestep 的通道排序/幅值差异是否真的大
  - 比较 `global scale` 与 `per-timestep scale` 的分布差异
- 只有当这个分析明显表明差异足够大，再考虑工程化的动态方案。

## 最推荐的落地路线

### 第一步：分析版，不改模型行为

- 增加一个离线分析脚本
- 输出每层 `x_span[t, c]`
- 输出每层 `scale[t, c]`
- 比较它们和全局 `scale[c]` 的偏差

这一步几乎没有模型风险。

### 第二步：尝试 stage-aware smooth

- 先把 timestep 离散成 3~4 个阶段
- 每个阶段一套 `scale`
- 仅在少数关键层上实验

### 第三步：如果必须做真正动态 smooth

则需要重新设计：

- 动态 smooth hook
- timestep-aware cache
- timestep-aware calibration objective
- 动态权重选择/重参数化

这已经接近一个新的量化机制，而不是在现有 smooth 上补一个小 feature。

## 关键代码锚点

- `deepcompressor/app/diffusion/dataset/calib_wan_loader.py`
  - `WanCalibDataset`
  - `WanCalibCacheLoader.iter_samples()`
  - `WanCalibCacheLoader.iter_layer_activations()`
- `deepcompressor/dataset/action.py`
  - `ConcatCacheAction.apply()`
- `deepcompressor/data/cache.py`
  - `TensorCache.get_standardized_data()`
  - `TensorsCache.extract()`
- `deepcompressor/calib/metric.py`
  - `ChannelMetric.abs_max()`
  - `ChannelMetric.root_mean_square()`
- `deepcompressor/calib/smooth.py`
  - `get_smooth_span()`
  - `get_smooth_scale()`
  - `ActivationSmoother`
  - `smooth_linear_modules()`
- `deepcompressor/calib/search.py`
  - `eval_kwargs`
  - `ipts.extract(i, eval_kwargs)`

## 一句话总结

- “按 timestep 分开算 scale”本身不难。
- 但“让模型在不同 timestep 真正使用不同 smooth_scale 且保持 smooth 的等价重参数化”需要动态权重或多权重分支，当前 DeepCompressor 架构并不直接支持。
