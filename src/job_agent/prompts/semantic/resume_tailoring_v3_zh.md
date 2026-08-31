# 目标简历改写

<!-- TASK -->
只生成有证据支持的中文简历表述。每条 bullet 的 requirement_id 和 evidence_level 必须
逐字复制对应 verified_evidence，不得推断或升级。每项要求最多生成一条简洁 bullet；
不得添加输入中不存在的数字、成果、上线经历或个人贡献，不得复现
forbidden_output_strings。

<!-- REPAIR -->
重新生成完整结果。公司、岗位、requirement_id 和 evidence_level 必须与输入逐字一致；
不得添加未提供事实或复现 forbidden_output_strings，所有面向用户的内容使用中文。
