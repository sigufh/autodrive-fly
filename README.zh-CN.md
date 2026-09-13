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

## 当前正式结果（v5 镜像未见评估）

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

正式 v5 在 `10000–10047` 的 48 条镜像训练道路上开启多巴胺学习与探索，随后由
全新引擎加载策略，在 `400–431` 的 32 条未见镜像道路上关闭学习和探索评估。
32/32 场景通关、每条通过 9 个障碍，首障碍前碰撞、出界、障碍碰撞和超时均为
0%。原始转向镜像误差为 0，执行转向镜像 MAE 为 `3.55e-8`，MDN 后退命令、
实际倒车和后退门控占比均为 0%。所有发布门槛通过，默认策略已更新为
`learned_v5`。

v5 的行为闭环要明确分层：MaleCNS DNp20 提供连接组驱动的转向残差，DNpe017
调整默认前进速度；复眼障碍奇分量选择空侧，工程道路回正项负责绕过障碍后回中。
后两项是显式工程控制，不是对果蝇生物网络自然实现路径规划的主张。道路安全层
仍不读取障碍，只使用位置、速度、航向与预测横向位置。

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

页面通过“驾驶场景”选择器切换，默认仍是已发布的“直路避障（v5 基准）”。

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
