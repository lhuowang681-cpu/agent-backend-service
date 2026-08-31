# 中文作答卡

<!-- TASK -->
为每道面试题恰好生成一张中文作答卡。requirement_id、question、evidence_level 必须逐字
复制对应题目；supporting_evidence 逐字复制同 requirement_id 的 proof，boundary 逐字
复制 risk。short_answer 使用“结论—个人动作—证据—边界”的 60 秒框架，不得添加不存在
的数字、成果、上线经历或个人贡献。practice_prompts 必须是针对本题的中文练习动作。

<!-- REPAIR -->
重新生成完整作答卡。公司、岗位、题目、requirement_id、evidence_level、proof 和 risk
必须与输入逐字一致；不得添加未提供事实，所有面向用户的内容使用中文。
