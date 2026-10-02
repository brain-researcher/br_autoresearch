# EP01–EP21 当前概念图 — 2026-10-02

21 张主概念图均已实际打开，按最新设计核对科学关系、图中文字和正常显示时的可读性。本轮用 **built-in image-gen** 重画或编辑 17 张；保留已核对生成记录的 EP02、EP03、EP05、EP12 四张，另补充 EP03 的符号约定 caption。

每张新版生成后均再次看图检查，必要时局部修正。旧图片保留为历史版本；以下只展示当前选定主图。GOAL/paper-plan 中的 caption 保留必要的科学边界，图内减少密集流程框和警告条。

这些是研究设计示意图，不是实验结果，也不等于论文最终排版验收。图像修订未读取科学 payload、运行实验或修改执行合同。

第二轮交叉复查再次实际查看全部 21 张图：补明 EP18 三个比较模型共有 image + linear-label baseline；使用 built-in image-gen 将 EP21 更新为 v3，去掉 content loss 端多余的反向箭头。其余图无需再次修改。提交仅包含设计、图像与相应说明，不包含其他线程的运行代码或任务记录；未 push。

## 快速打开

| EP | 当前图 | 本轮处理 |
| --- | --- | --- |
| EP01 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/ep01_conceptual_question-imagegen-v3.png) | image-gen 更新 |
| EP02 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode02_dopamine_learning_rate_causality/outputs/ep02_conceptual_question.png) | 保留；已复查 |
| EP03 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/outputs/ep03_conceptual_question.png) | 保留；已复查 |
| EP04 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry/outputs/ep04_conceptual_question-v2.png) | image-gen 更新 |
| EP05 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp/outputs/ep05_question_imagegen.png) | 保留；已复查 |
| EP06 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode06_lfp_regional_fingerprints/outputs/ep06_question_imagegen-v2.png) | image-gen 更新 |
| EP07 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode07_lfp_session_transfer/outputs/ep07_question_imagegen-v2.png) | image-gen 更新 |
| EP08 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode08_lfp_electrode_trial_policy/outputs/figures/ep08_episode_series-v6.png) | image-gen 更新 |
| EP09 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode09_local_global_projection/outputs/ep09_conceptual_question-v3.png) | image-gen 更新 |
| EP10 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/ep10_within_target_implementation-v2.png) | image-gen 更新 |
| EP11 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/outputs/ep11_conceptual_question-v2.png) | image-gen 更新 |
| EP12 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/ep12_conceptual_main_figure-v2.png) | 保留；已复查 |
| EP13 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode13_bold_target_identifiability/outputs/ep13_conceptual_question-v2.png) | image-gen 更新 |
| EP14 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode14_openbhb_roi_site_generalization/outputs/ep14_conceptual_question-v4.png) | image-gen 更新 |
| EP15 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry/outputs/ep15_conceptual_question-v2.png) | image-gen 更新 |
| EP16 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode16_braingate_cross_session_failure_modes/outputs/ep16_question_imagegen-v2.png) | image-gen 更新 |
| EP17 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/ep17_conceptual_question-v3.png) | image-gen 更新 |
| EP18 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability/outputs/ep18_conceptual_question-v2.png) | image-gen 更新 |
| EP19 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/outputs/ep19_question_imagegen-v2.png) | image-gen 更新 |
| EP20 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/ep20_question_imagegen-v2.png) | image-gen 更新 |
| EP21 | [打开原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/ep21_neural_code_conversion_design-v3.png) | image-gen 更新 |

## EP01

重画：choice/RT、认知规格引起的条件 gain/loss map 位移，以及同一参与者池的团队分析生态。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/ep01_conceptual_question-imagegen-v3-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/ep01_conceptual_question-imagegen-v3.png)

![EP01 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode01_narps_deep_search/outputs/ep01_conceptual_question-imagegen-v3.png)

## EP02

保留：protocol contrast 与 learning-rule discrimination 分开；图像生成记录已核对。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode02_dopamine_learning_rate_causality/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode02_dopamine_learning_rate_causality/outputs/ep02_conceptual_question.prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode02_dopamine_learning_rate_causality/outputs/ep02_conceptual_question.png)

![EP02 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode02_dopamine_learning_rate_causality/outputs/ep02_conceptual_question.png)

## EP03

保留图像、更新 caption：极性作用于整个 prototype + residual；符号翻转是实现约束。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/outputs/ep03_conceptual_question.prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/outputs/ep03_conceptual_question.png)

![EP03 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode03_condition_semantics_effect_maps/outputs/ep03_conceptual_question.png)

## EP04

重画：native VLM/DINOv2 分别对齐 fMRI；突出 within-category OOD 比较。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry/outputs/ep04_conceptual_question-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry/outputs/ep04_conceptual_question-v2.png)

![EP04 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode04_shared_scene_geometry/outputs/ep04_conceptual_question-v2.png)

## EP05

保留：同一目标下的 reach 偏差与正确/错误 trial 配对；条件性 spike follow-up。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp/outputs/ep05_question_imagegen_prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp/outputs/ep05_question_imagegen.png)

![EP05 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode05_sensorimotor_lfp/outputs/ep05_question_imagegen.png)

## EP06

重画：common-time、direction-contrast、trial-residual 三成分及跨动物区域规则比较。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode06_lfp_regional_fingerprints/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode06_lfp_regional_fingerprints/outputs/ep06_question_imagegen-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode06_lfp_regional_fingerprints/outputs/ep06_question_imagegen-v2.png)

![EP06 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode06_lfp_regional_fingerprints/outputs/ep06_question_imagegen-v2.png)

## EP07

重画：history 的额外 trial-specific 配对收益；不再从未检出配对优势推断 mean/geometry-only。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode07_lfp_session_transfer/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode07_lfp_session_transfer/outputs/ep07_question_imagegen-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode07_lfp_session_transfer/outputs/ep07_question_imagegen-v2.png)

![EP07 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode07_lfp_session_transfer/outputs/ep07_question_imagegen-v2.png)

## EP08

局部编辑：相同观测点数、真正替换 electrode；保留开放探索，不固定 pilot 数量。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode08_lfp_electrode_trial_policy/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode08_lfp_electrode_trial_policy/outputs/figures/ep08_episode_series-v6-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode08_lfp_electrode_trial_policy/outputs/figures/ep08_episode_series-v6.png)

![EP08 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode08_lfp_electrode_trial_policy/outputs/figures/ep08_episode_series-v6.png)

## EP09

重画：固定 recipient target vector，比较 own/donor dendritic residual；去掉等效结果的生物原因归因。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode09_local_global_projection/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode09_local_global_projection/outputs/ep09_conceptual_question-v3-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode09_local_global_projection/outputs/ep09_conceptual_question-v3.png)

![EP09 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode09_local_global_projection/outputs/ep09_conceptual_question-v3.png)

## EP10

重画：共同 target A 内的 terminal layout、可重叠的 co-target context 与 pooling 后果。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/ep10_within_target_implementation-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/ep10_within_target_implementation-v2.png)

![EP10 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode10_single_cell_coprojection/outputs/ep10_within_target_implementation-v2.png)

## EP11

重画：flexible continuum 与可复用 axon-allocation patterns；预测新组分布而非宣称 cell types。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/outputs/ep11_conceptual_question-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/outputs/ep11_conceptual_question-v2.png)

![EP11 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode11_projection_types_vs_gradients/outputs/ep11_conceptual_question-v2.png)

## EP12

保留：cell-type average 丢失 input–output pairing；结构示意不等于信号流。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/ep12_conceptual_main_figure-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/ep12_conceptual_main_figure-v2.png)

![EP12 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode12_malecns_type_sufficiency/outputs/ep12_conceptual_main_figure-v2.png)

## EP13

重画：压缩观测与单一 scalar target；经典线性可估性，移除密集证明/警告框。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode13_bold_target_identifiability/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode13_bold_target_identifiability/outputs/ep13_conceptual_question-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode13_bold_target_identifiability/outputs/ep13_conceptual_question-v2.png)

![EP13 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode13_bold_target_identifiability/outputs/ep13_conceptual_question-v2.png)

## EP14

重画：ROI 表征与 unseen-domain age transfer；site 是泛化环境，不是删除对象。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode14_openbhb_roi_site_generalization/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode14_openbhb_roi_site_generalization/outputs/ep14_conceptual_question-v4-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode14_openbhb_roi_site_generalization/outputs/ep14_conceptual_question-v4.png)

![EP14 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode14_openbhb_roi_site_generalization/outputs/ep14_conceptual_question-v4.png)

## EP15

重画：四类统计解释、未见 Task B 的 18-condition geometry；不预定赢家或生物机制。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry/outputs/ep15_conceptual_question-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry/outputs/ep15_conceptual_question-v2.png)

![EP15 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode15_mdtb_topography_vs_geometry/outputs/ep15_conceptual_question-v2.png)

## EP16

重画：local signal、transport loss、recording-change emulator 和 32-label repair 的可共存表现。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode16_braingate_cross_session_failure_modes/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode16_braingate_cross_session_failure_modes/outputs/ep16_question_imagegen-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode16_braingate_cross_session_failure_modes/outputs/ep16_question_imagegen-v2.png)

![EP16 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode16_braingate_cross_session_failure_modes/outputs/ep16_question_imagegen-v2.png)

## EP17

重画：response estimate、voxel support、positive weighting 三种测量操作；不包装成训练因果效应。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/ep17_conceptual_question-v3-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/ep17_conceptual_question-v3.png)

![EP17 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode17_cneuromod_model_ranking_stability/outputs/ep17_conceptual_question-v3.png)

## EP18

局部编辑：10 Hz 是图像呈现率；共同 image base、三种 sharing basis；图内预测得分改为问号。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability/outputs/ep18_conceptual_question-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability/outputs/ep18_conceptual_question-v2.png)

![EP18 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode18_things_eeg_concept_stability/outputs/ep18_conceptual_question-v2.png)

## EP19

重画：300–600 ms 预测、sensor/context 增量、过去非重叠输入与始终保留的 older complement。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/outputs/ep19_question_imagegen-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/outputs/ep19_question_imagegen-v2.png)

![EP19 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode19_causal_motor_forecasting_benchmark/outputs/ep19_question_imagegen-v2.png)

## EP20

重画：修正 D0–D3 表格；scan + software 为联合 time-focused 操作；模型分歧要求新测量。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/ep20_question_imagegen-v2-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/ep20_question_imagegen-v2.png)

![EP20 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode20_neurocam_prior_guided_codesign/outputs/ep20_question_imagegen-v2.png)

## EP21

重画：不共享训练图像的 converter/decoder、共同测试与 readouts；去掉 blanket leakage-free 声明。

[设计与 caption](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/GOAL.md) · [Image-gen prompt](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/ep21_neural_code_conversion_design-v3-prompt.md) · [原图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/ep21_neural_code_conversion_design-v3.png)

![EP21 当前概念图](/oak/stanford/groups/russpold/users/zijiao/br_autoresearch/episode21_wang_neural_code_conversion_reproduction/outputs/ep21_neural_code_conversion_design-v3.png)
