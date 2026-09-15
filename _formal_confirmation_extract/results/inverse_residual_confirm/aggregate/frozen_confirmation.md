# CA-OPEX frozen confirmation and adapter controls

Evidence scope: Frozen confirmation template. The selected adapter and CA-OPEX versions were not run or tuned on the 50-episode 39300/49300 block; base seeds 1/10 were predeclared rather than performance-screened. External baseline arms on these episode seeds already existed and all such raw records are retained, so the whole evaluation block is not described as blind. The template remains unusable until final implementation SHA256 placeholders are resolved before any predeclared adapter training directory or OPEX confirmation output exists.

> Episode confidence intervals are paired fixed-checkpoint rollout intervals. They are not training-seed confidence intervals.

## Runs

| Run | Method | Label | Train seed | Status | Updates | Wall s (cumulative) | Total Q1 forward rows | Q1 backward rows | Frozen Q scale |
|---|---|---|---:|---|---:|---:|---:|---:|---:|
| inverse_p001_k8_u5k_seed1 | inverse_sampled_main | confirmation | 1 | complete | 5000 | 48.5694 | 20512768 | 10240000 | 23.2334 |
| inverse_p001_k8_u5k_seed10 | inverse_sampled_main | confirmation | 10 | complete | 5000 | 50.8966 | 20512768 | 10240000 | 23.3533 |
| identity_p001_k8_u5k_seed1 | direct_sampled_main | confirmation | 1 | complete | 5000 | 39.2379 | 20512768 | 10240000 | 23.1974 |
| identity_p001_k8_u5k_seed10 | direct_sampled_main | confirmation | 10 | complete | 5000 | 40.7088 | 20512768 | 10240000 | 23.3206 |
| inverse_qmean_p001_k8_u5k_seed1 | inverse_q1_at_channel_mean | confirmation | 1 | complete | 5000 | 56.7009 | 20512768 | 10240000 | 23.3515 |
| inverse_qmean_p001_k8_u5k_seed10 | inverse_q1_at_channel_mean | confirmation | 10 | complete | 5000 | 62.5184 | 20512768 | 10240000 | 23.4497 |
| identity_wide_p064_d2_k8_u5k_seed1 | direct_sampled_wide | confirmation | 1 | complete | 5000 | 41.4493 | 20512768 | 10240000 | 23.1974 |
| identity_wide_p064_d2_k8_u5k_seed10 | direct_sampled_wide | confirmation | 10 | complete | 5000 | 39.4030 | 20512768 | 10240000 | 23.3206 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed1 | direct_sampled_nominal_beta0 | confirmation | 1 | complete | 5000 | 27.6490 | 20512768 | 10240000 | 23.3074 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed10 | direct_sampled_nominal_beta0 | confirmation | 10 | complete | 5000 | 39.5204 | 20512768 | 10240000 | 23.4205 |

Adapter costs distinguish base-actor preprocessing, frozen Q-scale calibration, optimizer Q1 forwards, and optimizer Q1 backward rows. Wall and optimizer time are cumulative across resume invocations. Preprocessing seconds cover only the latest invocation; preprocessing rows are multiplied by the recorded invocation count because every resume recomputes them. State-complete resume is implemented, but uninterrupted-versus-resumed bitwise equality is verified only on CPU, not CUDA.

| Run | Invocations | Base actor rows/invocation | Base actor rows lifetime | Q-scale Q1 rows/invocation | Q-scale Q1 rows lifetime | Optimizer Q1 forward | Optimizer Q1 backward | Adapter MLP forward | Adapter MLP backward | Preprocess s (latest invocation) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| inverse_p001_k8_u5k_seed1 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 9.9503 |
| inverse_p001_k8_u5k_seed10 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 10.0485 |
| identity_p001_k8_u5k_seed1 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 0.2049 |
| identity_p001_k8_u5k_seed10 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 0.2063 |
| inverse_qmean_p001_k8_u5k_seed1 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 10.0756 |
| inverse_qmean_p001_k8_u5k_seed10 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 10.5390 |
| identity_wide_p064_d2_k8_u5k_seed1 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 0.2148 |
| identity_wide_p064_d2_k8_u5k_seed10 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 0.2088 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed1 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 0.2072 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed10 | 1 | 899940 | 899940 | 32768 | 32768 | 20480000 | 10240000 | 1280000 | 1280000 | 0.2109 |

## Evaluation means

| Run/evaluation | Label | Arm | Episodes | Raw mean | Normalized mean | Clip | Bound | Residual | Env/base-actor rows | Adapter MLP rows |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inverse_p001_k8_u5k_seed1/confirm_beta125 | confirmation | adapted | 50 | 1425.9210 | 31.0258 | 0.3829 | 0.2651 | 0.1016 | 22974 | 22974 |
| inverse_p001_k8_u5k_seed1/confirm_beta125 | confirmation | baseline_only | 50 | 1023.1147 | 22.2513 | 0.3870 | 0.3569 | 0.0000 | 16582 | 0 |
| inverse_p001_k8_u5k_seed1/confirm_clean | confirmation | adapted | 50 | 3471.3711 | 75.5825 | 0.0000 | 0.2311 | 0.1221 | 47331 | 47331 |
| inverse_p001_k8_u5k_seed1/confirm_clean | confirmation | baseline_only | 50 | 3737.6677 | 81.3833 | 0.0000 | 0.3091 | 0.0000 | 49546 | 0 |
| inverse_p001_k8_u5k_seed10/confirm_beta125 | confirmation | adapted | 50 | 1130.2334 | 24.5847 | 0.3814 | 0.2575 | 0.1001 | 18344 | 18344 |
| inverse_p001_k8_u5k_seed10/confirm_beta125 | confirmation | baseline_only | 50 | 868.6581 | 18.8868 | 0.3855 | 0.3511 | 0.0000 | 14493 | 0 |
| inverse_p001_k8_u5k_seed10/confirm_clean | confirmation | adapted | 50 | 3631.1598 | 79.0632 | 0.0000 | 0.1950 | 0.1339 | 50000 | 50000 |
| inverse_p001_k8_u5k_seed10/confirm_clean | confirmation | baseline_only | 50 | 3315.2810 | 72.1823 | 0.0000 | 0.2776 | 0.0000 | 44876 | 0 |
| identity_p001_k8_u5k_seed1/confirm_beta125 | confirmation | adapted | 50 | 1086.2795 | 23.6273 | 0.3491 | 0.1354 | 0.1485 | 18187 | 18187 |
| identity_p001_k8_u5k_seed1/confirm_beta125 | confirmation | baseline_only | 50 | 574.4708 | 12.4784 | 0.3203 | 0.0000 | 0.0000 | 10921 | 0 |
| identity_p001_k8_u5k_seed1/confirm_clean | confirmation | adapted | 50 | 3757.4093 | 81.8133 | 0.0000 | 0.1004 | 0.1703 | 50000 | 50000 |
| identity_p001_k8_u5k_seed1/confirm_clean | confirmation | baseline_only | 50 | 3651.1302 | 79.4982 | 0.0000 | 0.0000 | 0.0000 | 47675 | 0 |
| identity_p001_k8_u5k_seed10/confirm_beta125 | confirmation | adapted | 50 | 809.2230 | 17.5921 | 0.3463 | 0.1348 | 0.1467 | 13966 | 13966 |
| identity_p001_k8_u5k_seed10/confirm_beta125 | confirmation | baseline_only | 50 | 460.1882 | 9.9889 | 0.3175 | 0.0000 | 0.0000 | 9358 | 0 |
| identity_p001_k8_u5k_seed10/confirm_clean | confirmation | adapted | 50 | 3034.3083 | 66.0618 | 0.0000 | 0.0506 | 0.1500 | 50000 | 50000 |
| identity_p001_k8_u5k_seed10/confirm_clean | confirmation | baseline_only | 50 | 3567.9810 | 77.6869 | 0.0000 | 0.0000 | 0.0000 | 50000 | 0 |
| inverse_qmean_p001_k8_u5k_seed1/confirm_beta125 | confirmation | adapted | 50 | 1299.6890 | 28.2760 | 0.3807 | 0.2438 | 0.1047 | 20631 | 20631 |
| inverse_qmean_p001_k8_u5k_seed1/confirm_beta125 | confirmation | baseline_only | 50 | 1023.1147 | 22.2513 | 0.3870 | 0.3569 | 0.0000 | 16582 | 0 |
| inverse_qmean_p001_k8_u5k_seed10/confirm_beta125 | confirmation | adapted | 50 | 1286.7327 | 27.9938 | 0.3792 | 0.2473 | 0.1016 | 20592 | 20592 |
| inverse_qmean_p001_k8_u5k_seed10/confirm_beta125 | confirmation | baseline_only | 50 | 868.6581 | 18.8868 | 0.3855 | 0.3511 | 0.0000 | 14493 | 0 |
| identity_wide_p064_d2_k8_u5k_seed1/confirm_beta125 | confirmation | adapted | 50 | 874.7399 | 19.0192 | 0.3402 | 0.1037 | 0.1533 | 15700 | 15700 |
| identity_wide_p064_d2_k8_u5k_seed1/confirm_beta125 | confirmation | baseline_only | 50 | 574.4708 | 12.4784 | 0.3203 | 0.0000 | 0.0000 | 10921 | 0 |
| identity_wide_p064_d2_k8_u5k_seed10/confirm_beta125 | confirmation | adapted | 50 | 666.7116 | 14.4877 | 0.3371 | 0.1032 | 0.1475 | 12184 | 12184 |
| identity_wide_p064_d2_k8_u5k_seed10/confirm_beta125 | confirmation | baseline_only | 50 | 460.1882 | 9.9889 | 0.3175 | 0.0000 | 0.0000 | 9358 | 0 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed1/confirm_beta125 | confirmation | adapted | 50 | 683.6669 | 14.8570 | 0.3356 | 0.1197 | 0.1739 | 12792 | 12792 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed1/confirm_beta125 | confirmation | baseline_only | 50 | 574.4708 | 12.4784 | 0.3203 | 0.0000 | 0.0000 | 10921 | 0 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed10/confirm_beta125 | confirmation | adapted | 50 | 654.5024 | 14.2217 | 0.3368 | 0.1187 | 0.1704 | 12831 | 12831 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed10/confirm_beta125 | confirmation | baseline_only | 50 | 460.1882 | 9.9889 | 0.3175 | 0.0000 | 0.0000 | 9358 | 0 |

Adapter rollout base-actor and adapter-MLP rows are derived exactly from each arm's raw episode lengths. The evaluator records only one paired total wall time, so no per-arm wall time is imputed.

## Cross-file baseline integrity

| Check | Semantics | Episodes | Endpoints | Status |
|---|---|---:|---|---|
| seed1_calibrated_inverse_baseline_exact | same frozen base actor followed by the same calibrated clipped-uniform-mean inverse; value estimator and residual adapter are bypassed | 50 | inverse_p001_k8_u5k_seed1/confirm_beta125/baseline_only; inverse_qmean_p001_k8_u5k_seed1/confirm_beta125/baseline_only; external_ca_opex_inverse_seed1/confirm_beta125/baseline_only | verified_elementwise_identical |
| seed10_calibrated_inverse_baseline_exact | same frozen base actor followed by the same calibrated clipped-uniform-mean inverse; value estimator and residual adapter are bypassed | 50 | inverse_p001_k8_u5k_seed10/confirm_beta125/baseline_only; inverse_qmean_p001_k8_u5k_seed10/confirm_beta125/baseline_only; external_ca_opex_inverse_seed10/confirm_beta125/baseline_only | verified_elementwise_identical |
| seed1_identity_baseline_exact | same frozen base actor used directly as command; adapter delta, model beta, and OPEX gradient path are bypassed | 50 | identity_p001_k8_u5k_seed1/confirm_beta125/baseline_only; identity_wide_p064_d2_k8_u5k_seed1/confirm_beta125/baseline_only; identity_nominal_beta0_p064_d2_k8_u5k_seed1/confirm_beta125/baseline_only; external_opex_original_seed1/confirm_beta125/baseline_only; external_ca_opex_identity_seed1/confirm_beta125/baseline_only; external_base_identity_seed1/confirm_beta125/external | verified_elementwise_identical |
| seed10_identity_baseline_exact | same frozen base actor used directly as command; adapter delta, model beta, and OPEX gradient path are bypassed | 50 | identity_p001_k8_u5k_seed10/confirm_beta125/baseline_only; identity_wide_p064_d2_k8_u5k_seed10/confirm_beta125/baseline_only; identity_nominal_beta0_p064_d2_k8_u5k_seed10/confirm_beta125/baseline_only; external_opex_original_seed10/confirm_beta125/baseline_only; external_ca_opex_identity_seed10/confirm_beta125/baseline_only; external_base_identity_seed10/confirm_beta125/external | verified_elementwise_identical |

## Strict external controls

| Run/evaluation | Method | Schema | Seed | Arm | Episodes | Raw mean | Normalized mean | Beta | Arm env steps | Deployment env steps | Paired eval env steps | Q1 forward rows | Q1 backward rows | Cost scope |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| external_base_identity_seed1/confirm_beta125 | td3bc_base_identity | evaluate_td3bc_v1 | 1 | external | 50 | 574.4708 | 12.4784 | 1.2500 | 10921 | 58596 | — | — | — | rollout_only_no_online_critic |
| external_base_identity_seed1/confirm_clean | td3bc_base_identity | evaluate_td3bc_v1 | 1 | external | 50 | 3651.1302 | 79.4982 | 0.0000 | 47675 | 58596 | — | — | — | rollout_only_no_online_critic |
| external_base_identity_seed10/confirm_beta125 | td3bc_base_identity | evaluate_td3bc_v1 | 10 | external | 50 | 460.1882 | 9.9889 | 1.2500 | 9358 | 59358 | — | — | — | rollout_only_no_online_critic |
| external_base_identity_seed10/confirm_clean | td3bc_base_identity | evaluate_td3bc_v1 | 10 | external | 50 | 3567.9810 | 77.6869 | 0.0000 | 50000 | 59358 | — | — | — | rollout_only_no_online_critic |
| external_scalar12_seed1/confirm_beta125 | td3bc_scalar_1p2 | evaluate_td3bc_v1 | 1 | external | 50 | 979.1241 | 21.2931 | 1.2500 | 16143 | 66143 | — | — | — | rollout_only_no_online_critic |
| external_scalar12_seed1/confirm_clean | td3bc_scalar_1p2 | evaluate_td3bc_v1 | 1 | external | 50 | 3820.9740 | 83.1980 | 0.0000 | 50000 | 66143 | — | — | — | rollout_only_no_online_critic |
| external_scalar12_seed10/confirm_beta125 | td3bc_scalar_1p2 | evaluate_td3bc_v1 | 10 | external | 50 | 766.6970 | 16.6657 | 1.2500 | 13252 | 63252 | — | — | — | rollout_only_no_online_critic |
| external_scalar12_seed10/confirm_clean | td3bc_scalar_1p2 | evaluate_td3bc_v1 | 10 | external | 50 | 3756.9560 | 81.8034 | 0.0000 | 50000 | 63252 | — | — | — | rollout_only_no_online_critic |
| external_oracle_inverse_seed1/confirm_beta125 | td3bc_oracle_beta1p25_inverse | evaluate_td3bc_v1 | 1 | external | 50 | 1011.2683 | 21.9933 | 1.2500 | 16493 | 66191 | — | — | — | rollout_only_no_online_critic |
| external_oracle_inverse_seed1/confirm_clean | td3bc_oracle_beta1p25_inverse | evaluate_td3bc_v1 | 1 | external | 50 | 3752.6607 | 81.7099 | 0.0000 | 49698 | 66191 | — | — | — | rollout_only_no_online_critic |
| external_oracle_inverse_seed10/confirm_beta125 | td3bc_oracle_beta1p25_inverse | evaluate_td3bc_v1 | 10 | external | 50 | 847.3211 | 18.4220 | 1.2500 | 14239 | 59435 | — | — | — | rollout_only_no_online_critic |
| external_oracle_inverse_seed10/confirm_clean | td3bc_oracle_beta1p25_inverse | evaluate_td3bc_v1 | 10 | external | 50 | 3331.7761 | 72.5416 | 0.0000 | 45196 | 59435 | — | — | — | rollout_only_no_online_critic |
| external_dual_identity_seed1/confirm_beta125 | td3bc_dual_executed_commanded_identity | evaluate_td3bc_v1 | 1 | external | 50 | 739.5802 | 16.0750 | 1.2500 | 12945 | 62945 | — | — | — | rollout_only_no_online_critic |
| external_dual_identity_seed1/confirm_clean | td3bc_dual_executed_commanded_identity | evaluate_td3bc_v1 | 1 | external | 50 | 3460.7057 | 75.3501 | 0.0000 | 50000 | 62945 | — | — | — | rollout_only_no_online_critic |
| external_dual_identity_seed10/confirm_beta125 | td3bc_dual_executed_commanded_identity | evaluate_td3bc_v1 | 10 | external | 50 | 673.2915 | 14.6310 | 1.2500 | 12070 | 62070 | — | — | — | rollout_only_no_online_critic |
| external_dual_identity_seed10/confirm_clean | td3bc_dual_executed_commanded_identity | evaluate_td3bc_v1 | 10 | external | 50 | 3237.1204 | 70.4797 | 0.0000 | 50000 | 62070 | — | — | — | rollout_only_no_online_critic |
| external_opex_original_seed1/confirm_beta125 | original_structure_opex_t1 | channel_opex_v1 | 1 | adapted | 50 | 851.1230 | 18.5048 | 1.2500 | 14497 | 14497 | 25418 | 14497 | 14497 | adapted_arm_deployment_controller_only |
| external_opex_original_seed1/confirm_beta125 | original_structure_opex_t1 | channel_opex_v1 | 1 | baseline_only | 50 | 574.4708 | 12.4784 | 1.2500 | 10921 | 14497 | 25418 | 14497 | 14497 | adapted_arm_deployment_controller_only |
| external_opex_original_seed10/confirm_beta125 | original_structure_opex_t1 | channel_opex_v1 | 10 | adapted | 50 | 744.9876 | 16.1928 | 1.2500 | 13093 | 13093 | 22451 | 13093 | 13093 | adapted_arm_deployment_controller_only |
| external_opex_original_seed10/confirm_beta125 | original_structure_opex_t1 | channel_opex_v1 | 10 | baseline_only | 50 | 460.1882 | 9.9889 | 1.2500 | 9358 | 13093 | 22451 | 13093 | 13093 | adapted_arm_deployment_controller_only |
| external_ca_opex_inverse_seed1/confirm_beta125 | channel_aware_opex_inverse_anchor | channel_opex_v1 | 1 | adapted | 50 | 1434.4251 | 31.2110 | 1.2500 | 22431 | 22431 | 39013 | 358896 | 358896 | adapted_arm_deployment_controller_only |
| external_ca_opex_inverse_seed1/confirm_beta125 | channel_aware_opex_inverse_anchor | channel_opex_v1 | 1 | baseline_only | 50 | 1023.1147 | 22.2513 | 1.2500 | 16582 | 22431 | 39013 | 358896 | 358896 | adapted_arm_deployment_controller_only |
| external_ca_opex_inverse_seed10/confirm_beta125 | channel_aware_opex_inverse_anchor | channel_opex_v1 | 10 | adapted | 50 | 1273.2322 | 27.6997 | 1.2500 | 20192 | 20192 | 34685 | 323072 | 323072 | adapted_arm_deployment_controller_only |
| external_ca_opex_inverse_seed10/confirm_beta125 | channel_aware_opex_inverse_anchor | channel_opex_v1 | 10 | baseline_only | 50 | 868.6581 | 18.8868 | 1.2500 | 14493 | 20192 | 34685 | 323072 | 323072 | adapted_arm_deployment_controller_only |
| external_ca_opex_identity_seed1/confirm_beta125 | channel_aware_opex_identity_anchor | channel_opex_v1 | 1 | adapted | 50 | 991.4510 | 21.5616 | 1.2500 | 17093 | 17093 | 28014 | 273488 | 273488 | adapted_arm_deployment_controller_only |
| external_ca_opex_identity_seed1/confirm_beta125 | channel_aware_opex_identity_anchor | channel_opex_v1 | 1 | baseline_only | 50 | 574.4708 | 12.4784 | 1.2500 | 10921 | 17093 | 28014 | 273488 | 273488 | adapted_arm_deployment_controller_only |
| external_ca_opex_identity_seed10/confirm_beta125 | channel_aware_opex_identity_anchor | channel_opex_v1 | 10 | adapted | 50 | 1024.6537 | 22.2849 | 1.2500 | 17384 | 17384 | 26742 | 278144 | 278144 | adapted_arm_deployment_controller_only |
| external_ca_opex_identity_seed10/confirm_beta125 | channel_aware_opex_identity_anchor | channel_opex_v1 | 10 | baseline_only | 50 | 460.1882 | 9.9889 | 1.2500 | 9358 | 17384 | 26742 | 278144 | 278144 | adapted_arm_deployment_controller_only |

## Paired comparisons

> The six predeclared primary comparisons are flagged below and summarized first at checkpoint-seed level in the next table. Other episode-level intervals are secondary descriptions; no multiple-comparison correction is applied.

| Comparison | Primary? | Label | Pairs | Positive | Raw Δ | Raw t 95% | Raw bootstrap 95% | Norm Δ | Norm t 95% | Norm bootstrap 95% |
|---|---|---|---:|---:|---:|---|---|---:|---|---|
| inverse_p001_k8_u5k_seed1_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 29 | 402.8063 | [115.1388, 690.4738] | [133.1628, 686.6220] | 8.7745 | [2.5081, 15.0408] | [2.9007, 14.9569] |
| inverse_p001_k8_u5k_seed1_clean_adapted_minus_own_baseline | False | confirmation | 50 | 4 | -266.2966 | [-425.4497, -107.1434] | [-424.4475, -120.0854] | -5.8008 | [-9.2677, -2.3339] | [-9.2459, -2.6159] |
| inverse_p001_k8_u5k_seed10_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 31 | 261.5753 | [39.1948, 483.9559] | [54.1825, 480.8035] | 5.6980 | [0.8538, 10.5422] | [1.1803, 10.4735] |
| inverse_p001_k8_u5k_seed10_clean_adapted_minus_own_baseline | False | confirmation | 50 | 18 | 315.8788 | [112.9195, 518.8380] | [130.7302, 520.7883] | 6.8809 | [2.4598, 11.3020] | [2.8477, 11.3445] |
| identity_p001_k8_u5k_seed1_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 45 | 511.8087 | [335.2624, 688.3549] | [346.8283, 689.9755] | 11.1489 | [7.3031, 14.9946] | [7.5551, 15.0299] |
| identity_p001_k8_u5k_seed1_clean_adapted_minus_own_baseline | False | confirmation | 50 | 11 | 106.2792 | [-31.8331, 244.3915] | [-14.5094, 251.9215] | 2.3151 | [-0.6934, 5.3237] | [-0.3161, 5.4877] |
| identity_p001_k8_u5k_seed10_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 41 | 349.0348 | [207.0846, 490.9851] | [215.6193, 492.6676] | 7.6031 | [4.5110, 10.6953] | [4.6969, 10.7319] |
| identity_p001_k8_u5k_seed10_clean_adapted_minus_own_baseline | False | confirmation | 50 | 0 | -533.6727 | [-544.0215, -523.3240] | [-543.9119, -523.8551] | -11.6252 | [-11.8506, -11.3997] | [-11.8482, -11.4113] |
| inverse_qmean_p001_k8_u5k_seed1_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 24 | 276.5743 | [-7.4687, 560.6173] | [17.7617, 565.2119] | 6.0247 | [-0.1627, 12.2121] | [0.3869, 12.3122] |
| inverse_qmean_p001_k8_u5k_seed10_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 31 | 418.0746 | [151.2012, 684.9480] | [160.8823, 681.4054] | 9.1070 | [3.2937, 14.9204] | [3.5045, 14.8433] |
| identity_wide_p064_d2_k8_u5k_seed1_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 34 | 300.2690 | [107.3661, 493.1720] | [121.0062, 492.5465] | 6.5409 | [2.3388, 10.7429] | [2.6359, 10.7293] |
| identity_wide_p064_d2_k8_u5k_seed10_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 42 | 206.5234 | [119.6392, 293.4077] | [121.0207, 291.4696] | 4.4988 | [2.6061, 6.3914] | [2.6362, 6.3492] |
| identity_nominal_beta0_p064_d2_k8_u5k_seed1_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 27 | 109.1961 | [-23.0980, 241.4901] | [-19.8821, 235.5752] | 2.3787 | [-0.5032, 5.2605] | [-0.4331, 5.1316] |
| identity_nominal_beta0_p064_d2_k8_u5k_seed10_beta125_adapted_minus_own_baseline | False | confirmation | 50 | 38 | 194.3142 | [89.7023, 298.9262] | [98.8277, 301.8778] | 4.2328 | [1.9540, 6.5116] | [2.1528, 6.5759] |
| seed1_beta125_inverse_sampled_minus_direct_sampled | False | confirmation | 50 | 27 | 339.6414 | [-4.6926, 683.9755] | [12.2847, 681.3513] | 7.3985 | [-0.1022, 14.8992] | [0.2676, 14.8421] |
| seed1_clean_inverse_sampled_minus_direct_sampled | False | confirmation | 50 | 2 | -286.0382 | [-421.0633, -151.0132] | [-423.8226, -168.4212] | -6.2309 | [-9.1722, -3.2896] | [-9.2323, -3.6688] |
| seed1_beta125_inverse_qmean_minus_inverse_sampled | False | confirmation | 50 | 24 | -126.2320 | [-466.6254, 214.1614] | [-460.5623, 195.5223] | -2.7498 | [-10.1646, 4.6651] | [-10.0326, 4.2591] |
| seed1_beta125_direct_wide_minus_direct_main | False | confirmation | 50 | 13 | -211.5396 | [-450.8212, 27.7419] | [-444.9897, 16.7845] | -4.6080 | [-9.8204, 0.6043] | [-9.6933, 0.3656] |
| seed1_beta125_direct_nominal_q_minus_direct_channel_q | False | confirmation | 50 | 9 | -402.6126 | [-604.8025, -200.4227] | [-606.9909, -214.7560] | -8.7702 | [-13.1746, -4.3659] | [-13.2223, -4.6781] |
| seed10_beta125_inverse_sampled_minus_direct_sampled | False | confirmation | 50 | 37 | 321.0104 | [94.5217, 547.4991] | [108.6678, 544.0775] | 6.9927 | [2.0590, 11.9263] | [2.3671, 11.8518] |
| seed10_clean_inverse_sampled_minus_direct_sampled | False | confirmation | 50 | 50 | 596.8515 | [584.1109, 609.5921] | [584.7012, 609.4731] | 13.0014 | [12.7239, 13.2789] | [12.7367, 13.2763] |
| seed10_beta125_inverse_qmean_minus_inverse_sampled | False | confirmation | 50 | 29 | 156.4993 | [-112.6683, 425.6668] | [-104.0246, 418.7074] | 3.4091 | [-2.4543, 9.2724] | [-2.2660, 9.1208] |
| seed10_beta125_direct_wide_minus_direct_main | False | confirmation | 50 | 21 | -142.5114 | [-264.9061, -20.1167] | [-262.2614, -27.8294] | -3.1044 | [-5.7705, -0.4382] | [-5.7129, -0.6062] |
| seed10_beta125_direct_nominal_q_minus_direct_channel_q | False | confirmation | 50 | 15 | -154.7206 | [-300.8475, -8.5937] | [-295.9082, -15.0997] | -3.3703 | [-6.5535, -0.1872] | [-6.4459, -0.3289] |
| seed1_beta125_inverse_adapter_minus_oracle_inverse | False | confirmation | 50 | 32 | 414.6527 | [124.8868, 704.4185] | [144.7059, 706.1196] | 9.0325 | [2.7204, 15.3446] | [3.1522, 15.3816] |
| seed1_beta125_inverse_adapter_minus_scalar12 | False | confirmation | 50 | 30 | 446.7968 | [118.4294, 775.1643] | [138.9443, 772.4205] | 9.7327 | [2.5798, 16.8856] | [3.0267, 16.8259] |
| seed1_beta125_inverse_adapter_minus_base_identity | False | confirmation | 50 | 43 | 851.4501 | [581.7480, 1121.1522] | [599.3404, 1122.6513] | 18.5474 | [12.6724, 24.4224] | [13.0556, 24.4551] |
| seed1_beta125_direct_adapter_minus_base_identity | False | confirmation | 50 | 45 | 511.8087 | [335.2624, 688.3549] | [346.8283, 689.9755] | 11.1489 | [7.3031, 14.9946] | [7.5551, 15.0299] |
| seed1_clean_inverse_adapter_minus_scalar12 | False | confirmation | 50 | 1 | -349.6029 | [-483.9495, -215.2563] | [-487.5210, -231.4549] | -7.6155 | [-10.5420, -4.6890] | [-10.6198, -5.0419] |
| seed1_clean_direct_adapter_minus_base_identity | False | confirmation | 50 | 11 | 106.2792 | [-31.8331, 244.3915] | [-14.5094, 251.9215] | 2.3151 | [-0.6934, 5.3237] | [-0.3161, 5.4877] |
| seed1_beta125_inverse_adapter_minus_dual_policy | False | confirmation | 50 | 42 | 686.3408 | [421.9365, 950.7450] | [438.2880, 949.8471] | 14.9508 | [9.1912, 20.7104] | [9.5474, 20.6908] |
| seed1_beta125_direct_adapter_minus_dual_policy | False | confirmation | 50 | 36 | 346.6993 | [151.5008, 541.8978] | [163.0137, 540.3119] | 7.5523 | [3.3002, 11.8043] | [3.5510, 11.7698] |
| seed10_beta125_inverse_adapter_minus_oracle_inverse | False | confirmation | 50 | 30 | 282.9123 | [57.4010, 508.4237] | [70.3192, 506.2259] | 6.1628 | [1.2504, 11.0751] | [1.5318, 11.0273] |
| seed10_beta125_inverse_adapter_minus_scalar12 | False | confirmation | 50 | 36 | 363.5364 | [146.8792, 580.1937] | [159.6441, 575.8156] | 7.9190 | [3.1995, 12.6385] | [3.4776, 12.5432] |
| seed10_beta125_inverse_adapter_minus_base_identity | False | confirmation | 50 | 42 | 670.0453 | [446.3211, 893.7694] | [461.4180, 891.0475] | 14.5958 | [9.7223, 19.4693] | [10.0512, 19.4100] |
| seed10_beta125_direct_adapter_minus_base_identity | False | confirmation | 50 | 41 | 349.0348 | [207.0846, 490.9851] | [215.6193, 492.6676] | 7.6031 | [4.5110, 10.6953] | [4.6969, 10.7319] |
| seed10_clean_inverse_adapter_minus_scalar12 | False | confirmation | 50 | 2 | -125.7963 | [-142.4847, -109.1078] | [-141.7992, -109.6340] | -2.7403 | [-3.1038, -2.3767] | [-3.0889, -2.3882] |
| seed10_clean_direct_adapter_minus_base_identity | False | confirmation | 50 | 0 | -533.6727 | [-544.0215, -523.3240] | [-543.9119, -523.8551] | -11.6252 | [-11.8506, -11.3997] | [-11.8482, -11.4113] |
| seed10_beta125_inverse_adapter_minus_dual_policy | False | confirmation | 50 | 37 | 456.9419 | [238.0773, 675.8065] | [252.0160, 674.9433] | 9.9537 | [5.1861, 14.7213] | [5.4897, 14.7025] |
| seed10_beta125_direct_adapter_minus_dual_policy | False | confirmation | 50 | 26 | 135.9315 | [-8.9641, 280.8270] | [0.6612, 280.9574] | 2.9610 | [-0.1953, 6.1173] | [0.0144, 6.1202] |
| seed1_beta125_original_opex_effect | False | confirmation | 50 | 35 | 276.6521 | [107.7155, 445.5888] | [119.9353, 444.2308] | 6.0264 | [2.3464, 9.7064] | [2.6126, 9.6768] |
| seed10_beta125_original_opex_effect | False | confirmation | 50 | 37 | 284.7994 | [119.6368, 449.9620] | [137.0735, 456.4919] | 6.2039 | [2.6061, 9.8017] | [2.9859, 9.9439] |
| seed1_beta125_ca_opex_inverse_minus_ca_opex_identity | True | confirmation | 50 | 37 | 442.9741 | [195.9006, 690.0476] | [207.6924, 685.2024] | 9.6494 | [4.2674, 15.0315] | [4.5242, 14.9260] |
| seed10_beta125_ca_opex_inverse_minus_ca_opex_identity | True | confirmation | 50 | 30 | 248.5784 | [-0.2218, 497.3786] | [17.1659, 494.8046] | 5.4149 | [-0.0048, 10.8346] | [0.3739, 10.7785] |
| seed1_beta125_ca_opex_inverse_minus_original_opex | True | confirmation | 50 | 40 | 583.3021 | [287.9638, 878.6404] | [296.1251, 864.2627] | 12.7062 | [6.2728, 19.1397] | [6.4506, 18.8265] |
| seed10_beta125_ca_opex_inverse_minus_original_opex | True | confirmation | 50 | 39 | 528.2446 | [296.0510, 760.4382] | [310.8661, 759.7847] | 11.5069 | [6.4490, 16.5649] | [6.7717, 16.5506] |
| seed1_beta125_ca_opex_inverse_effect | True | confirmation | 50 | 29 | 411.3104 | [111.9889, 710.6320] | [126.3294, 703.1488] | 8.9597 | [2.4395, 15.4799] | [2.7519, 15.3169] |
| seed10_beta125_ca_opex_inverse_effect | True | confirmation | 50 | 30 | 404.5741 | [163.1038, 646.0443] | [180.4292, 647.9839] | 8.8130 | [3.5529, 14.0730] | [3.9303, 14.1152] |
| seed1_beta125_ca_opex_identity_effect | False | confirmation | 50 | 39 | 416.9801 | [240.9216, 593.0386] | [252.7051, 590.4924] | 9.0832 | [5.2481, 12.9183] | [5.5048, 12.8629] |
| seed10_beta125_ca_opex_identity_effect | False | confirmation | 50 | 44 | 564.4656 | [385.2604, 743.6707] | [399.0986, 743.9895] | 12.2959 | [8.3922, 16.1996] | [8.6937, 16.2066] |
| seed1_beta125_inverse_adapter_minus_original_opex | False | confirmation | 50 | 40 | 574.7980 | [263.0296, 886.5664] | [276.9556, 880.1522] | 12.5210 | [5.7297, 19.3123] | [6.0330, 19.1726] |
| seed10_beta125_inverse_adapter_minus_original_opex | False | confirmation | 50 | 33 | 385.2458 | [159.1433, 611.3484] | [170.4997, 606.7435] | 8.3919 | [3.4667, 13.3172] | [3.7140, 13.2169] |
| seed1_beta125_inverse_adapter_minus_ca_opex_inverse | False | confirmation | 50 | 28 | -8.5041 | [-357.4502, 340.4419] | [-347.9754, 333.6037] | -0.1852 | [-7.7864, 7.4160] | [-7.5801, 7.2670] |
| seed10_beta125_inverse_adapter_minus_ca_opex_inverse | False | confirmation | 50 | 21 | -142.9987 | [-416.5607, 130.5633] | [-407.0761, 119.4027] | -3.1150 | [-9.0741, 2.8441] | [-8.8675, 2.6010] |
| seed1_beta125_inverse_adapter_minus_ca_opex_identity | False | confirmation | 50 | 35 | 434.4700 | [123.6437, 745.2962] | [134.2170, 738.7712] | 9.4642 | [2.6934, 16.2350] | [2.9237, 16.0929] |
| seed10_beta125_inverse_adapter_minus_ca_opex_identity | False | confirmation | 50 | 27 | 105.5797 | [-150.7813, 361.9407] | [-134.4890, 361.8503] | 2.2999 | [-3.2845, 7.8843] | [-2.9296, 7.8823] |

## Primary cross-controller checkpoint summary

> Each row is descriptive over exactly two base-policy checkpoint seeds (1 and 10). The same 50 rollout-case seed pairs are crossed with both checkpoints; episodes are not pooled into n=100 and no training-seed confidence interval or significance claim is made.

| Comparison family | Checkpoint seeds | n checkpoints | Per-checkpoint raw Δ | Mean raw Δ | Mean normalized Δ | Positive checkpoints | Training-seed CI |
|---|---|---:|---|---:|---:|---:|---|
| ca_opex_inverse_minus_ca_opex_identity | [1,10] | 2 | seed1=442.9741; seed10=248.5784 | 345.7763 | 7.5322 | 2 | — |
| ca_opex_inverse_minus_original_opex | [1,10] | 2 | seed1=583.3021; seed10=528.2446 | 555.7733 | 12.1066 | 2 | — |
| ca_opex_inverse_minus_inverse_only | [1,10] | 2 | seed1=411.3104; seed10=404.5741 | 407.9423 | 8.8863 | 2 | — |

## External-control checkpoint evidence

| Method/evaluation/arm | Checkpoint seeds | n checkpoints | Per-checkpoint rollout means | Descriptive raw mean | Descriptive normalized mean | Training-seed CI |
|---|---|---:|---|---:|---:|---|
| channel_aware_opex_identity_anchor/confirm_beta125/adapted | [1,10] | 2 | seed1=991.4510; seed10=1024.6537 | 1008.0524 | 21.9232 | — |
| channel_aware_opex_identity_anchor/confirm_beta125/baseline_only | [1,10] | 2 | seed1=574.4708; seed10=460.1882 | 517.3295 | 11.2337 | — |
| channel_aware_opex_inverse_anchor/confirm_beta125/adapted | [1,10] | 2 | seed1=1434.4251; seed10=1273.2322 | 1353.8286 | 29.4554 | — |
| channel_aware_opex_inverse_anchor/confirm_beta125/baseline_only | [1,10] | 2 | seed1=1023.1147; seed10=868.6581 | 945.8864 | 20.5690 | — |
| original_structure_opex_t1/confirm_beta125/adapted | [1,10] | 2 | seed1=851.1230; seed10=744.9876 | 798.0553 | 17.3488 | — |
| original_structure_opex_t1/confirm_beta125/baseline_only | [1,10] | 2 | seed1=574.4708; seed10=460.1882 | 517.3295 | 11.2337 | — |
| td3bc_base_identity/confirm_beta125/external | [1,10] | 2 | seed1=574.4708; seed10=460.1882 | 517.3295 | 11.2337 | — |
| td3bc_base_identity/confirm_clean/external | [1,10] | 2 | seed1=3651.1302; seed10=3567.9810 | 3609.5556 | 78.5926 | — |
| td3bc_dual_executed_commanded_identity/confirm_beta125/external | [1,10] | 2 | seed1=739.5802; seed10=673.2915 | 706.4359 | 15.3530 | — |
| td3bc_dual_executed_commanded_identity/confirm_clean/external | [1,10] | 2 | seed1=3460.7057; seed10=3237.1204 | 3348.9131 | 72.9149 | — |
| td3bc_oracle_beta1p25_inverse/confirm_beta125/external | [1,10] | 2 | seed1=1011.2683; seed10=847.3211 | 929.2947 | 20.2076 | — |
| td3bc_oracle_beta1p25_inverse/confirm_clean/external | [1,10] | 2 | seed1=3752.6607; seed10=3331.7761 | 3542.2184 | 77.1257 | — |
| td3bc_scalar_1p2/confirm_beta125/external | [1,10] | 2 | seed1=979.1241; seed10=766.6970 | 872.9106 | 18.9794 | — |
| td3bc_scalar_1p2/confirm_clean/external | [1,10] | 2 | seed1=3820.9740; seed10=3756.9560 | 3788.9650 | 82.5007 | — |

## Frozen-Q audits

| Run/audit | Label | States | K | Beta | Q1 gain mean/median | Q2 gain mean/median | Min-twin mean/median | Disagreement Δ mean/median | Physical rows | Critic rows |
|---|---|---:|---:|---:|---|---|---|---|---:|---:|
| inverse_p001_k8_u5k_seed1/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.2463/0.1772 | 0.2300/0.1593 | 0.2365/0.1672 | 0.0032/-0.0011 | 12807680 | 25615360 |
| inverse_p001_k8_u5k_seed10/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.2500/0.1811 | 0.2118/0.1430 | 0.2351/0.1672 | -0.0085/-0.0075 | 12807680 | 25615360 |
| identity_p001_k8_u5k_seed1/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.3792/0.3153 | 0.3549/0.2870 | 0.3631/0.2981 | 0.0078/0.0006 | 12807680 | 25615360 |
| identity_p001_k8_u5k_seed10/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.3852/0.3244 | 0.3279/0.2665 | 0.3613/0.3014 | -0.0096/-0.0109 | 12807680 | 25615360 |
| inverse_qmean_p001_k8_u5k_seed1/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.2379/0.1713 | 0.2195/0.1507 | 0.2270/0.1603 | 0.0035/-0.0023 | 12807680 | 25615360 |
| inverse_qmean_p001_k8_u5k_seed10/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.2417/0.1751 | 0.2042/0.1378 | 0.2273/0.1618 | -0.0088/-0.0086 | 12807680 | 25615360 |
| identity_wide_p064_d2_k8_u5k_seed1/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.4851/0.3166 | 0.4725/0.3064 | 0.4716/0.3083 | 0.0144/0.0014 | 12807680 | 25615360 |
| identity_wide_p064_d2_k8_u5k_seed10/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.5018/0.3249 | 0.4384/0.2811 | 0.4700/0.3047 | 0.0002/-0.0049 | 12807680 | 25615360 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed1/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.5501/0.3454 | 0.5387/0.3360 | 0.5335/0.3344 | 0.0217/0.0043 | 12807680 | 25615360 |
| identity_nominal_beta0_p064_d2_k8_u5k_seed10/heldout_k64 | confirmation | 100060 | 64 | 1.2500 | 0.5874/0.3526 | 0.5056/0.3035 | 0.5424/0.3307 | 0.0083/-0.0027 | 12807680 | 25615360 |

## Statistical-unit audit

| Method | Label | Unit | Seeds | n | Multiple seeds (descriptive only) | Training-seed CI |
|---|---|---|---|---:|---|---|
| direct_sampled_main | confirmation | full_pipeline_training_seed | [1,10] | 2 | True | — |
| direct_sampled_nominal_beta0 | confirmation | full_pipeline_training_seed | [1,10] | 2 | True | — |
| direct_sampled_wide | confirmation | full_pipeline_training_seed | [1,10] | 2 | True | — |
| inverse_q1_at_channel_mean | confirmation | full_pipeline_training_seed | [1,10] | 2 | True | — |
| inverse_sampled_main | confirmation | full_pipeline_training_seed | [1,10] | 2 | True | — |
