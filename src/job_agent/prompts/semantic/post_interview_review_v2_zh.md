# 真实面试 AI 复盘

<!-- TASK -->
用中文诊断这次真实面试暴露的准备缺口，逐题覆盖 question_ids，区分 knowledge、
expression、project_evidence、truth_boundary。没有回答且没有反馈时 status 必须为
insufficient_information。只能引用输入中存在的 requirement_id 和 evidence_id，不得
升级证据、编造面试官反馈、预测通过率或录用概率。improved_answer 只能使用
verified_evidence 支持的事实；信息不足时留空并给出补录动作。

<!-- REPAIR -->
重新生成完整复盘。company、title、round_id 必须逐字一致；逐题覆盖且只能引用当前
question/requirement/evidence ID；缺少回答时必须 abstain；不得预测面试结果或添加输入
中没有的经历。
