# JD 结构化

<!-- TASK -->
从岗位 JD 中提取明确的必备要求和加分项。公司、岗位、地点和 raw_jd 必须与输入逐字一致；
requirement_id 必须唯一。不要把推测写成岗位明确要求，也不要把 forbidden_output_strings
复制到 raw_jd 之外的字段。所有面向用户的内容使用中文。

<!-- REPAIR -->
重新生成完整结果。逐字复制公司、岗位和 raw_jd；为每项要求生成唯一 ID；不得遗漏明确
要求，不得在 raw_jd 之外复现 forbidden_output_strings。
