# AutoDrive Fly：MaleCNS 果蝇全脑视觉驾驶实验

AutoDrive Fly 是一个受 [Doomfly](https://github.com/nftechie/doomfly) 启发的闭环研究项目：将模拟道路的视觉刺激映射到雄性果蝇完整中枢神经系统（MaleCNS v1.0），通过持续神经动力学、真实结构连接上的局部可塑性和多巴胺式奖励调制，产生车辆转向与速度控制。

本项目不是传统自动驾驶栈，也没有隐藏的目标检测器、A*、规则避障器或大语言模型。
原始动作来自神经状态和冻结校准的结构增益；可关闭的道路安全层只在临近边缘时
限制执行动作，并在界面中显式显示介入量。

## 系统概览

```text
二维道路与圆形障碍
  ↓ 48 × 24 灰度刺激
3,344 个映射后的 R1–R6 光感受器
  ↓ 每个车辆动作前运行 4 个全脑微步
166,700 个 MaleCNS 神经元 / 25,582,938 条有向连接
  ↓
双侧 DNp20 → 转向，双侧 DNpe017 → 油门
  ↓
车辆运动、碰撞与通关回报
  ↓
双侧 PPL101 奖励预测误差与对手式训练调制
  ↓
1,571 条真实运动输入连接上的资格迹可塑性
```

前端同时展示：

- 车辆俯视道路、历史轨迹和短时轨迹投影；
- 48 × 24 复眼刺激；
- 实时速度、转向、任务回报和双侧 PPL101 调制信号；
- MaleCNS 三维神经活动和方向性活动连接；
- 覆盖 330 个类别连接组和全部 25,582,938 条边的完整拓扑示意；
- 单步、连续运行、新场景、在线实验、探索、安全约束和恢复发布策略控制。

## 与 Doomfly 的关系

本项目参考的是 Doomfly 中值得保留的方法论，而不是复制其结论或参数。

| 方法 | Doomfly | AutoDrive Fly |
|---|---|---|
| 连接组 | MaleCNS 全连接结构 | MaleCNS v1.0 全连接结构 |
| 视觉输入 | R1–R6 亮度、R8 色彩代理 | 3,344 个 R1–R6 的 optic-hex 位置代理 |
| 动力学 | 持续 LIF 风格神经时间步 | 持续有界稀疏动力学，每个车辆动作 4 个 CNS 微步 |
| 动作读出 | 指定降行神经元控制 Doom | 双侧 DNp20 / DNpe017 控制转向和油门 |
| 可塑性 | KC→MBON11 既有连接、资格迹和 DAN 调制 | DNp20/DNpe017 真实输入连接、中心化活动、资格迹和双侧 PPL101 调制 |
| 学习验证 | Doomfly v6 文档报告未证明生存学习 | 匹配暴露、未见种子、冻结对照和部署前新引擎加载门槛；当前结果同样为负 |

Doomfly v6 的公开实验本身是负结果：学习开启后虽然改变了突触，但没有证明视觉条件化或生存提升。因此，本项目没有把“权重发生变化”当作学习成功，而是要求未见场景上的距离和原始回报都获得正收益。

## 当前正式结果（行为稳定性）

当前发布的是**冻结校准策略**，不是学习后策略。它保存完整归一化状态和
双侧运动零点，不更新突触。正式安全约束消融固定如下：

- 同一冻结校准检查点；
- 评估种子 `200–231`；
- 学习和探索均关闭；
- 唯一变量是道路安全约束开关；
- 安全层不读取障碍，只在接近道路边缘或航向向外时限制神经转向。

| 指标 | 约束关闭 | 约束开启 | 改善 |
|---|---:|---:|---:|
| 平均前进距离 | 26.80 m | 30.33 m | **+3.53 m** |
| 平均原始回报 | 4.38 | 5.21 | **+0.84** |
| 侧边界退出率 | 25% | 0% | **−25 个百分点** |
| 最大横向偏移 | 3.08 m | 2.41 m | **−0.67 m** |
| 障碍碰撞率 | 75% | 100% | 未改善 |
| 安全层介入率 | 0% | 26.2% | 显式记录 |

配对 bootstrap 95% 区间：

- 距离增益：`[1.07, 6.47] m`；
- 原始回报增益：`[0.25, 1.53]`。

两项区间都完全高于 0。该结果证明行为约束解决了“容易转出道路”的问题，
但没有证明果蝇网络学会避障：32 个场景的通关率仍是 0%。
安全层平均动作修正量为 `0.032`，并非全程覆盖神经命令。

稳定化后的在线塑性对照是负结果：学习策略相对冻结策略距离 `−1.21m`、
回报 `−0.29`，因此评估器拒绝发布学习检查点。这取代了旧版 `+41.79m`
的结论；旧检查点遗漏运行均值/方差，旧“重放误差 0”没有经过新引擎加载验证。

完整证据：

- [`artifacts/driving-evaluation.json`](artifacts/driving-evaluation.json)：在线塑性负结果；
- [`artifacts/behavior-constraint-ablation.json`](artifacts/behavior-constraint-ablation.json)：道路约束开关消融；
- [`artifacts/deployed-stable-policy.json`](artifacts/deployed-stable-policy.json)：最终部署策略表现；
- [`docs/completion-audit.md`](docs/completion-audit.md)：目标—证据完成审计；
- [`artifacts/checkpoints/driving-policy.npz`](artifacts/checkpoints/driving-policy.npz)：v2 冻结校准策略。

策略 SHA-256：

```text
3a6d12ffba3af2bf6562d02a0d72e68acdd702e4a377e838ba7568ab4ece030e
```

v2 检查点保存 1,571 条既有结构连接的增益、源神经元身份、运行均值/方差、
奖励基线和运动零点。加载时验证格式、body ID、形状和 `0.97–1.03×` 增益范围。

## 快速开始

### 1. 环境

推荐 Python 3.11 或 3.12、Node.js 20+。

```bash
git clone https://github.com/sigufh/autodrive-fly.git
cd autodrive-fly
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
npm_config_cache=.npm-cache npm --prefix apps/web install
```

### 2. 下载并校验 MaleCNS

MaleCNS 原始连接数据超过 1GB，不进入 Git。URL、大小和 SHA-256 均固定在 [`configs/data-manifest.yaml`](configs/data-manifest.yaml)。

```bash
.venv/bin/autodrive-fly download --dataset malecns
.venv/bin/autodrive-fly verify --dataset malecns
make prepare
```

默认不会下载可选的约 12.7GB 突触点表和 6.8GB partner 表。驾驶闭环不需要它们。`make prepare` 会构建：

- 166,700 节点的稳定 body ID 索引；
- 原始计数与目标归一化 CSR 图；
- 三维 soma/alternate-soma 概览；
- 完整类别通路和 40K 强连接；
- R1–R6 映射会在首次启动时由真实连接和 optic-hex 注释生成。

### 3. 启动

终端一：

```bash
./scripts/run-api.sh
```

终端二：

```bash
npm_config_cache=.npm-cache npm --prefix apps/web run dev -- --host 127.0.0.1 --port 5174
```

打开 <http://127.0.0.1:5174>。API 健康检查位于 <http://127.0.0.1:8000/api/health>。

仓库附带约 20KB 的 v2 冻结校准检查点。页面会明确显示检查点类型。点击
“恢复发布策略”会撤销当前 API 进程内的实验更新并重新加载冻结校准策略。

## 复现实验

正式评估会进行 48 组匹配训练暴露和 32 组未见测试；在完整 MaleCNS 图上运行需要较长时间。

```bash
make evaluate-driving
```

评估器只有在以下九个门槛全部通过时才发布实验学习检查点：

1. 平均距离增益大于 0；
2. 平均原始回报增益大于 0；
3. 距离增益配对 bootstrap 95% 区间下界大于 0；
4. 回报增益配对 bootstrap 95% 区间下界大于 0。
5. 侧边界退出率不劣于冻结基线；
6. 远障碍转向不劣于冻结基线；
7. 侧边界退出率不超过 10%；
8. 远障碍平均绝对转向不超过 0.15；
9. 转向变化率不劣于冻结基线。

复现冻结校准策略和道路约束消融：

```bash
make calibrate-policy
.venv/bin/autodrive-fly evaluate-constraints
```

开发验证：

```bash
make test
npm_config_cache=.npm-cache npm --prefix apps/web run test:browser
```

公开驾驶主线当前为 34 项 Python 测试、6 项前端测试和 2 项真实浏览器测试。

## 目录

```text
src/fly_emotion/driving/     道路环境、复眼映射、神经策略与评估
src/fly_emotion/connectome/  MaleCNS 图、动力学、骨架和通路资产
apps/api/                    FastAPI 状态与流式闭环服务
apps/web/                    React + Three.js 驾驶和全脑界面
configs/                     数据清单与固定校验值
artifacts/                   紧凑评估报告和已验证策略
docs/                        架构、实验、来源、边界和优化复盘
tests/                       后端、结构契约和真实数据测试
```

包内部仍沿用早期实验的 `fly_emotion` Python 命名空间；公开命令为
`autodrive-fly`。历史语言实验代码没有接入当前 API 或前端运行链路。

## 科学边界

- 复眼位置是从 R1–R6 的位置已知突触后伙伴推断的 optic-hex 代理，不是校准过的果蝇光转导模型。
- 神经活动是有界、持续、带递质符号先验的无量纲模型状态，不是实测膜电位。
- GABA、谷氨酸和组胺采用抑制性先验；项目没有完整的突触后受体数据，存在例外。
- DNp20/DNpe017 是真实神经元，但“转向/油门”是工程读出，不是已被生物实验确认的天然驾驶功能。
- PPL101 信号是奖励预测误差和训练期对手式教学变量，不是实测多巴胺浓度。训练期左右 clearance 是特权奖励塑形，不会在评估或运行时直接控制车辆。
- 当前正收益来自显式道路安全约束，不等价于多巴胺学习收益、真实自动驾驶能力、果蝇意识或生物学习机制验证。
- 当前测试通关率仍为 0%。项目只解决了侧边界退出和低频转向漂移，尚未解决障碍规避。

## 数据、许可证与引用

- MaleCNS v1.0：Janelia FlyEM，CC BY 4.0，<https://male-cns.janelia.org/download/>；
- Doomfly：<https://github.com/nftechie/doomfly>；
- Shiu et al., *Nature* 2024, “A Drosophila computational brain model reveals sensorimotor processing”；
- Lappalainen et al., *Nature* 2024, “Connectome-constrained networks predict neural activity across the fly visual system”；
- Huang et al., *Nature* 2024, “Dopamine-mediated interactions between short- and long-term memory dynamics”。

代码采用 Apache-2.0 许可证。MaleCNS 数据不包含在仓库中，使用时应遵循其数据许可和引用要求。

## 延伸阅读

- [架构说明](docs/architecture.md)
- [驾驶实验协议](docs/driving-experiment.md)
- [数据来源与计数口径](docs/data-provenance.md)
- [科学声明边界](docs/scientific-scope.md)
- [优化复盘](docs/optimization-review.zh-CN.md)
