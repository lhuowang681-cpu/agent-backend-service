# 简历证据映射

<!-- TASK -->
为每一项 must-have requirement 生成且只生成一条证据映射。只能引用简历中真实存在的内容；
没有支持证据时使用 C0 或 NONE，并明确风险。requirement_id 必须逐字复制，
evidence_id 必须唯一。不得复现 forbidden_output_strings。

<!-- REPAIR -->
重新生成完整映射，每个 must-have requirement_id 恰好一条，evidence_id 唯一。没有证据
时使用 C0/NONE，不得遗漏要求、升级证据或复现 forbidden_output_strings。
