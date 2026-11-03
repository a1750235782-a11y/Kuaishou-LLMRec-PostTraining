# 快手 LLM-Rec · 生成式推荐后训练实践

面向算法／大模型推荐岗位的项目展示：数据质量、推理监督与分阶段后训练。

仓库地址：[a1750235782-a11y/Kuaishou-LLMRec-PostTraining](https://github.com/a1750235782-a11y/Kuaishou-LLMRec-PostTraining)（私有）。

**参赛成绩：最好排名第 29 名，最高得分 1.079。** 以上为作者提供，尚未附榜单截图、队伍身份及轮次证明；最好排名与最高得分不假定来自同一次提交。作者已确认亲自实现并运行过三组方案。这里的代码为新编写的精简教学示例，原始参赛代码、模型及训练日志暂未归档，运行示例不能复现该比赛分数。

## 方法与展示范围

作者实施的技术路线涵盖自采样与目标 SID 损失 mask、数据清洗与分领域 Think/Nothink、分阶段 SFT/RL/DPO。方法来源参见[官方赛后报告 §7.2](https://arxiv.org/html/2609.39828#S7.SS2)，沿用公开方法并进行实施，不宣称原创提出。

| 模块 | 仓库可运行内容 | 边界 |
|---|---|---|
| 数据清洗与监督路由 | 精确短语清理；prod→think，其余→nothink | 规则演示，未接入语义重写或事实核查模型 |
| 自采样轨迹处理 | 对缓存合成轨迹按目标命中筛选、去重、限量 | 未运行真实模型自采样 |
| Masked RFT | 构造输入与 labels，屏蔽新增轨迹目标 SID，计算 causal CE | 使用人工 token ID 与 logits，无反向传播 |
| 偏好数据与 DPO | 同领域分层负例、排除已知正例及历史交互；标准 DPO 标量损失 | 未复现比赛的全部偏好优化细节 |
| RL | REINFORCE 标量目标、行为 F1 函数 | 未提供 PPO/GRPO、rollout 或训练器 |
| 评估 | SID 命中率、有效率、候选并集大小、模式 Jaccard | 未实现官方 PID 映射或综合评测 |

## 快速运行

在仓库根目录使用 Python 3.10+，仅依赖标准库，无需 GPU、模型下载或账号凭据。

```bash
python -m llmrec.demo
python -m unittest discover -s tests -v
```

演示读取 `examples/synthetic.json`，输出 `artifacts/demo.json`。固定输入下 SID 命中率应为 0.5，有效 SID 比例 0.6。它们是合成样例的确定性检查值，没有竞赛意义，不能与 1.079 比较。

## 目录

```text
llmrec/
  data.py          # 清理、路由、轨迹筛选、偏好对
  objectives.py    # labels、causal CE、DPO、REINFORCE、F1
  evaluation.py    # SID 级候选诊断
  demo.py          # 无依赖运行入口
examples/synthetic.json
tests/test_core.py
docs/
  METHODS.md
  REPRODUCIBILITY.md
  EXPERIMENTS.md
  RESUME.md
requirements.txt
.gitignore
```

## 数据与环境

仓库只包含人工合成的文本、标识与预测。`prod:1:2:3` 是演示格式，不是模型的 SID 特殊 token；代码不能直接读取官方数据或 checkpoint。真实数据、模型、用户行为、私钥及环境凭据不在上传范围。接入自有数据前需单独确认使用与再分发授权。

复现步骤及真实训练接入要求见 [REPRODUCIBILITY](docs/REPRODUCIBILITY.md)，方法细节见 [METHODS](docs/METHODS.md)，成绩证据与空白消融记录见 [EXPERIMENTS](docs/EXPERIMENTS.md)，可复制简历见 [RESUME](docs/RESUME.md)。

## 参考

- [官方赛后报告](https://arxiv.org/html/2609.39828)，提供比赛及领先方案背景。
- [DPO 原论文](https://arxiv.org/abs/2305.18290)，用于标准偏好损失的概念参考。
- [比赛网站](https://ks-llmrec.streamlake.com/)，数据、规则及成绩应以相应比赛阶段官方材料为准。

仓库初始按私有方式准备，尚未选择开源许可；参考资料与竞赛资产的权利归原权利人。
