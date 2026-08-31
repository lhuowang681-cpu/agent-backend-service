# Mock Interview Plan

- Mode: technical
- Company: ByteDance
- Role: LLM Post-training Intern
- Persona: fast-paced ownership press interviewer
- Difficulty: realistic

## Live Rules

- Ask one question at a time.
- Use at most 1-2 follow-ups per primary question.
- Score each answer on a 1-5 scale after the candidate answers.
- Give only one-line feedback during the mock.
- Do not provide the correct answer during the mock.
- Save teaching, model answers, and remediation for the debrief.

## Scoring Dimensions

- technical_depth
- evidence_quality
- ownership_clarity
- communication
- composure
- role_fit

## Question Queue

### Q1. 请结合一个真实项目说明你的经验：熟悉 SFT / LoRA 微调流程

- Requirement ID: `req_sft_lora`
- Focus: 追问数据格式、训练配置、loss 与 eval 的关系
- Time limit: 180 seconds
- Follow-ups:
  - What exact data, config, metric, or log can support this answer?
  - Which part was independently done by you?

### Q2. 在没有项目证据前，你会如何验证和补齐这项能力：了解 RLHF / DPO / GRPO / reward model 基本方法

- Requirement ID: `req_alignment`
- Focus: 追问偏好数据、reward/verifier 设计和算法差异
- Time limit: 180 seconds
- Risk flags: missing evidence, do not overclaim
- Follow-ups:
  - What exact data, config, metric, or log can support this answer?
  - Which part was independently done by you?

### Q3. 请结合一个真实项目说明你的经验：具备大模型评测、消融实验和 bad case 分析经验

- Requirement ID: `req_eval`
- Focus: 追问 baseline、指标定义、失败样例和局限性
- Time limit: 180 seconds
- Follow-ups:
  - What exact data, config, metric, or log can support this answer?
  - Which part was independently done by you?

## Debrief Boundary

- Do not write `15_mock_interview_debrief.md` until candidate answers are collected and scored.
- Use answer cards only after the live round, during debrief.
