# 示例验证记录

2026-11-03，Windows / Python 3.14.6，本地运行完成：

- `python -m unittest discover -s tests -v`：6 项通过，覆盖 mask 保留输入、原样本监督、因果移位、无效候选计分、负例排除、DPO 参考策略与轨迹去重。
- `python -m llmrec.demo`：成功生成 JSON；SID 命中率 0.5，有效率 0.6，与合成输入的预期一致。

这只证明精简示例在上述环境运行成功。尚未验证其他 Python 版本、GPU 训练、真实模型、官方综合评测或参赛成绩。没有进行耗时与性能基准测试。
