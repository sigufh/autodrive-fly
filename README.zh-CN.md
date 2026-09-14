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
默认前进 → DNp20 转向残差 / DNpe017 调速 / 四个 MDN 后退
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
| 动作读出 | 指定降行神经元控制 Doom | 默认前进，DNp20 转向残差、DNpe017 调速、MDN 后退 |
| 可塑性 | KC→MBON11 既有连接、资格迹和 DAN 调制 | DNp20/DNpe017 真实输入连接、中心化活动、资格迹和双侧 PPL101 调制 |
| 学习验证 | Doomfly v6 文档报告未证明生存学习 | 匹配暴露、未见种子、冻结对照和部署前新引擎加载门槛；当前结果同样为负 |

Doomfly v6 的公开实验本身是负结果：学习开启后虽然改变了突触，但没有证明视觉条件化或生存提升。因此，本项目没有把“权重发生变化”当作学习成功，而是要求未见场景上的距离和原始回报都获得正收益。

## 保留的工程避障基线（v5 镜像未见评估）

本次使用 `10000–10047` 共 48 次校准暴露，保存检查点后用新引擎评估
`400–431` 的 32 个未见场景，即 16 对独立道路。相邻种子共享障碍纵向位置、
半径和横向绝对值，左右严格镜像。测试关闭学习与探索，不根据测试结果调参。

| 指标 | v5 已发布策略 | 零转向直行基线 |
|---|---:|---:|
| 平均终点纵坐标 | 120.30 m | 49.15 m |
| 平均通过障碍数 | 9.00 / 9 | 3.19 / 9 |
| 首障碍通过率 | 100% | 81.25% |
| 通关率 | 100% | 0% |
| 平均距离 | 120.30m | 49.15m |

v5 在 `10000–10047` 的 48 条镜像训练道路上开启多巴胺学习与探索，随后由
全新引擎加载策略，在 `400–431` 的 32 条未见镜像道路上关闭学习和探索评估。
32/32 场景通关、每条通过 9 个障碍，首障碍前碰撞、出界、障碍碰撞和超时均为
0%。原始转向镜像误差为 0，执行转向镜像 MAE 为 `3.55e-8`，MDN 后退命令、
实际倒车和后退门控占比均为 0%。所有发布门槛通过，默认策略已更新为
`learned_v5`。

这份结果现在明确命名为“工程避障基线”：MaleCNS DNp20 提供连接组驱动的
转向残差，DNpe017 调整默认前进速度；复眼障碍奇分量选择空侧，工程道路回正项
负责绕过障碍后回中。后两项和道路安全层都是工程控制，因此该 100% 通关率不能
用来证明果蝇连接组自主学会了避障。

证据与复现：

```bash
.venv/bin/autodrive-fly calibrate-policy --episodes 48 --evaluation-start 400 --evaluation-seeds 32
.venv/bin/autodrive-fly evaluate-constraints --evaluation-start 400 --evaluation-seeds 32 \
  --checkpoint artifacts/checkpoints/driving-policy.npz \
  --output artifacts/behavior-constraint-ablation-v5.json
```

- `artifacts/stable-policy-calibration.json`：完整训练、镜像分组、逐步轨迹、直行基线与门槛。
- `artifacts/checkpoints/driving-policy.npz`：已发布且默认加载的 v5 检查点。
- 策略 SHA-256：`e9de899238198dcdd060dba17043edf3c9098743cfd85c33064889ac108268f9`。

完整协议、工程归因边界和 v4 负结果见 [驾驶实验](docs/driving-experiment.md)。

## 当前主线：旧随机障碍地图上的纯 MaleCNS 决策

前端已恢复为原来的随机障碍直路、复眼和全脑活动展示。默认模式是“工程避障
基线（v5）”，保留以前可用的随机避障运行；可切换到“纯 MaleCNS 决策实验”。
在后者中，DNp20、DNpe017、MDN 的读出经限幅动作适配器直接转换为方向盘、油门
和后退；障碍、道路边界和规则不再直接合成或修正动作，只作为视觉刺激、环境
回报、碰撞/通关结果。

固定随机道路 `400–407` 上的冻结对照证据：

| 指标 | 工程避障基线 | 纯 MaleCNS 决策 |
|---|---:|---:|
| 通关率 | 100% | 0% |
| 平均距离 | 120.33m | 27.43m |
| 平均通过障碍 | 9.00 | 1.25 |
| 障碍碰撞率 | 0% | 100% |
| 安全层介入 / 动作修正 | 4.23% / 0.0242 | 0% / 0 |

这是 v6 训练前的纯神经基线，当时尚未学会避障；它不能被工程基线的成功掩盖。
训练前报告见
[`artifacts/neural-decision-baseline.json`](artifacts/neural-decision-baseline.json)。

```bash
make evaluate-neural-decision
```

### v6 第一阶段训练结果

纯神经 v6 使用独立检查点训练，不覆盖 v5 工程基线。训练只接收环境回报：前向
进展、完整通过障碍、碰撞、出界和通关；不提供“障碍在左就应右转”的标签，也不
调用道路安全层或工程回正。相邻镜像道路共享探索序列，转向噪声只取镜像反号。

在 24 回合单障碍课程训练后，新引擎加载候选并在未见 `640–647` 测试：

- 通关率：冻结纯神经 `0%` → 学习后 `50%`；
- 平均距离：`23.86m` → `27.53m`；
- 首障碍通过率：`100%` → `75%`，但学习后不再全部通过后驶出道路；
- 相同动作幅度、时序打乱基线：`0%` 通关、`19.89m`；
- 神经模式的安全层介入和工程动作修正均为 0。

该 v6 还在未见三障碍 `700–707` 上零样本实现 `8/8` 通关、平均 `3/3` 障碍。
完整九障碍的稳定奖励训练没有通过门槛：通过后方向盘幅度虽从 `0.161` 降至
`0.128`，横向漂移却从 `0.876m` 增至 `0.980m`，通关率和出界率没有改善。
该候选未替换当前 v6。

独立 v6 检查点：`artifacts/checkpoints/driving-policy.neural-v6.npz`，SHA-256：
`6746a2b9bf80f2e2a1d6a5a8c6039ed074a931dbf31ef0fdc9da911c326e2437`。

可复现证据：

- `artifacts/neural-v6-single-curriculum.json`：单障碍训练、冻结基线、时序打乱对照和发布门槛；
- `artifacts/neural-v6-transfer.json`：冻结 v6 的三障碍与完整九障碍迁移；
- `artifacts/neural-v6-nine-curriculum.json`：被拒绝的九障碍稳定奖励训练候选。
- `artifacts/neural-v6-motor-adaptation.json`：冻结 v6 的 DNp20 运动适应器配对消融。

### 避障后持续转向修复

轨迹审计发现，车辆完整通过障碍后的 30 步内仍保持较大的 DNp20 转向输出，固定
执行器继续积分航向，形成大幅 S 形并驶出道路。简单降低方向盘增益、加入稳定惩罚
或延迟奖励都没有在未见道路上同时改善任务与漂移，因此没有采用。

最终采用环境无关的 DNp20 运动适应器：它只观察运动神经输出的时间序列，以每步
8% 的慢基线衰减持续恒定偏转，保留新的转向变化；它不读取障碍、道路、横向位置、
航向、奖励或路线。冻结 v6 在未见 `900–907` 上的唯一变量消融：

| 指标 | 无运动适应 | 8%/步适应 |
|---|---:|---:|
| 通关率 | 0% | 75% |
| 平均距离 | 46.02m | 107.38m |
| 平均通过障碍 | 3.25 | 8.0 |
| 出界率 | 100% | 25% |
| 通过后平均方向盘幅度 | 0.230 | 0.193 |
| 通过后 30 步横向漂移 | 2.00m | 0.88m |

两组安全层介入和动作修正均为 0，因此这不是代码根据道路替代神经回正。

### 近全景、光流与身体感觉消融

MaleCNS 注释中确认并接入了真实感觉入口：`6,753` 个 T4a/T5a/T4b/T5b
方向选择视觉神经元、`205` 个平衡器感觉神经元（左 104 / 右 101）和 `424` 个
本体感觉上行神经元（左 210 / 右 214）。这些感觉输入只进入原连接组，不直接
修改方向盘。近全景覆盖约 `330°`，保留约 `30°` 后盲区。

四档使用相同训练种子 `10000–10003` 和测试种子 `960–963` 的筛选消融：

| 感觉档位（学习后） | 通关率 | 平均障碍 | 通过后转向 | 通过后漂移 |
|---|---:|---:|---:|---:|
| 当前 143° 前视 | 100% | 9.0 | 0.133 | 0.607m |
| 近全景 | 50% | 7.0 | 0.267 | 0.915m |
| 近全景 + T4/T5 光流 | 50% | 5.5 | 0.178 | 0.746m |
| 近全景 + 光流 + 身体反馈 | 0% | 1.0 | 0.110 | 0.240m |

结果支持“视野和身体状态确实影响转向终止”，但没有任何新增档位同时超过 front
的任务能力和稳定性；身体反馈虽然最平滑，却几乎失去避障。因此新增通道暂不部署，
当前服务继续使用已验证的 `front + 8% DNp20 运动适应器`。该消融只有 2 对独立
测试道路，属于筛选证据，不是最终生物学结论。

证据：`artifacts/neural-v6-sensory-ablation.json`。

后续固定状态重放确认，负优化首先来自全景坐标分布：全景即使不加光流，也有 45%
步骤与 front 的 DNp20 转向符号不同。默认光流和身体反馈继续放大运动读出，但没有
造成全网数值饱和。相关证据为 `artifacts/neural-v6-sensory-gain-audit.json`；因此
修复顺序是先保持前视表征并隔离周边视觉，再重构局部光流和身体感觉，而不是直接
把当前全局信号调小后部署。

当前实验分支已将全景改为“24 列左周边 + 原 48 列前视 + 24 列右周边”，中央
143° 图像逐像素保持不变。在相同 `960–963` 道路上，周边实际增益 0.001 保持
100% 通关和 9/9 障碍，并把完整窗口每米漂移从 0.0706 降到 0.0638；它只作为
后续局部光流实验底座，尚未替换默认 front。证据：
`artifacts/neural-v6-peripheral-visual-ablation.json`。

T4/T5 光流现已改为 6 区域非循环估计，并通过真实上游连接和 optic-hex 注释为
6,752/6,753 个神经元推断局部位置。但冻结 v6 即使在绝对增益 0.0003 下仍退化为
50% 通关，因此实验默认光流增益为 0：保留编码，等待后续感觉通路学习，不部署。
证据：`artifacts/neural-v6-regional-flow-ablation.json`。

## 城市交通场景 Alpha

在不替换 v5 直路基准的前提下，项目新增独立的 `city` 场景，用于把“局部视觉
避障”推进到“地图路线 + 交通规则 + 局部避障”的分层问题。当前 Harbor Grid
Alpha 是可复现的合成城市道路图，不是现实城市数字孪生：

- 约 282.5m 的多路段路线，包含直路、右转、左转和三条命名道路；
- 两个信号化路口、停止线、红绿灯让行与违规记录；
- 双向车道宽度、合法车道目标、限速和被占用车道的前瞻换道任务；
- 两个同向静态路侧演员；复眼—MaleCNS—局部避障仍负责近场通行；
- 地图任务层只提供路线曲率、目标车道和红灯速度上限，不读取复眼射线、不替代
  障碍的空侧选择。

三个固定城市种子（`0, 1, 7`）的完整闭环 pilot 均到达终点（约 `282.8m`）、
通过 2 个演员、零红灯违规，平均转向变化约 `0.022`。这只是开发验证，不等同于
真实城市泛化或新的 v5 发布声明；后续需要独立训练/测试划分、动态交通参与者、
行人、优先级规则、地图格式（OpenDRIVE/OSM）和分布外评估。

城市 Alpha 保留为命令行/代码资产，当前页面主线已恢复为随机障碍直路，以便
对照工程基线和纯 MaleCNS 决策实验。

```bash
curl -X POST http://127.0.0.1:8000/api/driving/reset \
  -H 'content-type: application/json' \
  -d '{"seed":7,"keep_learning":false,"scenario":"city"}'
```

## 历史结果（v2，仅供追溯）

以下 v2 数值来自旧道路生成器与读出，不能作为 v4 的性能结论。

当时发布的是**冻结校准策略**，不是学习后策略。它保存完整归一化状态和
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

启动后加载已发布的 v5 `learned_v5` 检查点。点击“恢复发布策略”会撤销进程内
实验更新并恢复这份已验证策略。

## 复现实验

正式评估会进行 48 组匹配训练暴露和 32 组未见测试；在完整 MaleCNS 图上运行需要较长时间。

```bash
make evaluate-driving
```

评估器除以下历史九项门槛外，还要求通关率至少 50%、首障碍通过率至少 75%、
平均通过至少 4.5 个障碍、提前碰撞率不超过 25%、超时率不超过 10%；
冻结校准发布另外要求原始转向镜像误差不超过 `1e-6`。统计重采样单位是道路对。

1. 平均距离增益大于 0；
2. 平均原始回报增益大于 0；
3. 距离增益配对 bootstrap 95% 区间下界大于 0；
4. 回报增益配对 bootstrap 95% 区间下界大于 0。
5. 侧边界退出率不劣于冻结基线；
6. 远障碍转向不劣于冻结基线；
7. 侧边界退出率不超过 10%；
8. 远障碍平均绝对转向不超过 0.15；
9. 转向变化率不劣于冻结基线。

复现 v5 训练、未见评估和道路约束消融：

```bash
make calibrate-policy
.venv/bin/autodrive-fly evaluate-constraints
```

开发验证：

```bash
make test
npm_config_cache=.npm-cache npm --prefix apps/web run test:browser
```

公开驾驶主线当前为 49 项 Python 测试、6 项前端测试和 3 项真实浏览器测试。
全工作区还包含未纳入驾驶主线的旧语言实验测试，其 `scikit-learn` 声明断言
与已移除语言依赖的主项目不一致，本次没有引入该退役依赖。

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
- 直行为默认纵向原语；DNp20 仅在超过证据死区时产生转向残差，DNpe017 调速，
  四个真实 MDN 仅在显著活动时触发后退。以上车辆语义仍是工程映射。
- PPL101 信号是奖励预测误差和训练期对手式教学变量，不是实测多巴胺浓度。训练期左右 clearance 是特权奖励塑形，不会在评估或运行时直接控制车辆。
- 历史道路安全约束收益不适用于新版道路和读出，也不等价于多巴胺学习收益、真实自动驾驶能力或生物学习机制验证。
- 当前镜像测试通关率仍为 0%，候选不及直行基线；本次解决了固定侧偏与评估混淆，尚未解决障碍规避。

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
