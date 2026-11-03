# 实现说明

本仓库示例由项目整理时新编写，用于解释数据与损失处理。作者确认实施过相关路线，但尚未提供原训练日志核验具体配置。下面给出的实现选择均为示例选择。

## 数据流程

`clean_caption` 仅删除三条硬编码营销短语并压缩空白，保留其他字符；不能保证适合真实业务，应检查误删。监督路由使用显式领域枚举，未知领域直接报错。真实清洗应保留原文、变更理由与事实校验记录，按授权处理。

轨迹筛选函数一次只处理一个请求。目标命中、归一化文本精确去重、最多保留两条均在本地完成。两条是演示值，没有被确认是参赛参数。命中答案不能证明推理过程正确。自采样需自行接入训练集 checkpoint 的真实生成过程；验证／测试目标不能参与筛选。

## Token 监督

`build_sequence` 接收分段后的 token ID。Prompt 为 -100；原始 SFT 监督推理与答案；新增轨迹仅监督推理，目标 SID labels 置 -100，但 input_ids 保留 SID。`causal_cross_entropy` 使用 logits[t] 对 labels[t+1]，避免重复 shift。全 mask 会报错，防止无有效监督被当作正常更新。

接入真实 tokenizer 时必须从最终 chat template 获取可靠的字符／token span，检查特殊 SID token、截断、EOS、padding 与 packed sequence 边界。不能通过搜索普通字符串假定 SID 与 token 一一对应。Transformers 模型通常内部完成 labels shift，需根据所用模型核实；不要再手工移位。

## 偏好与强化

偏好对从同领域 catalog 选择不同一级码或同一级不同二级码的候选，排除所有已知正例与历史交互。这里按 catalog 顺序取各类首个候选，方便复现；不是训练用最优采样器。未交互不等于不喜欢，仍存在假负例。

DPO 函数计算标准目标 `softplus(-beta*((logp_chosen-logp_rejected)-(logref_chosen-logref_rejected)))`。输入为相同 prompt 下完成部分 log 概率之和；参考策略固定。生产实现需保持格式、推理前缀与评分范围一致，屏蔽 prompt/padding。标量示例没有梯度、label smoothing 或正例辅助生成损失。

REINFORCE 示例仅展示 `-mean(log_prob*(reward-baseline))`；奖励与基线在真实自动求导中应 detach。它不构成完整分阶段 RL：还需采样器、优化器、KL 约束、参考策略及稳定性监控。真实阶段顺序由作者训练记录确定，此处不编造 RL 算法型号、奖励权重或学习率。

## 评测约定

每模式先截取前 K 条原始候选，再过滤非法／跨领域候选，禁止用后续候选补位。模式内去重后求并集。SID 命中率按请求平均，任一目标命中即为 1；所有无效预测的请求仍计入分母。有效率按原始候选位置计数，重复合法候选仍合法；Jaccard 空并集取 0。

真实官方指标需 SID→PID 映射、阶段对应评分器及多任务权重，不能用本示例 SID 命中率推断综合得分。
