# CA-OPEX continuation results

This report records the post-confirmation continuation experiments. The formal
confirmation bundle is preserved separately and is not overwritten.

## Completed experiments

- Walker2d-v4: training seeds 1, 2, 3, 4, and 10; five controller arms; 50
  common-random-number paired rollouts per checkpoint.
- Hopper-v4: training seeds 2, 3, and 4; the same five controller arms and
  paired-rollout protocol.
- Walker2d channel severity: beta in {0.5, 0.9, 1.1, 1.4}, with an independent
  512-pair calibration for each beta and three checkpoints per beta.

## Main conclusions

Walker2d supports the complete CA-OPEX mechanism under the frozen
K=8, T=2, eta=0.1, delta=0.25 configuration. Across five training seeds,
complete CA-OPEX improves over calibrated identity by 6.16 normalized-return
points (95% t interval [0.01, 12.31], positive on 4/5 seeds), over
inverse-only by 10.13 points, over tuned nominal-Q by 11.40 points, and over
original OPEX by 12.63 points. The latter three comparisons are positive on
all five seeds and their training-seed intervals exclude zero.

Hopper provides a qualified cross-task result. Complete CA-OPEX improves over
original OPEX by 2.25 normalized-return points (3/3 positive; 95% t interval
[0.69, 3.80]), but does not improve over inverse-only: the mean difference is
-0.57 with a 95% interval [-2.41, 1.27]. The stronger claim about the
expected-Q correction therefore does not transfer cleanly to Hopper.

The beta grid is complete and independently verified. At beta=1.4 the
complete-vs-inverse-only difference is +3.31 normalized points with a
three-checkpoint descriptive 95% interval [2.01, 4.60]. At beta=0.5 the same
difference is +4.78 but the interval crosses zero [-14.92, 24.47]. These
intervals are descriptive over three checkpoints, not five-seed inference.

## Reproduction and provenance

The raw records, checkpoints, calibration files, aggregation scripts, verifier,
tests, logs, and `results/ca_opex_scaleup_bundle_v3.tgz` are included in this
repository. The scale-up bundle SHA-256 is
`e2f889d2e69e7a12d964aa7a185dd797bce3d3ceb4f3359c9baad61e2e47766b`.

Run the aggregate scripts first and then the corresponding independent
verifiers. The formal v3 bundle remains a separate prerequisite for reproducing
the original confirmation; the scale-up bundle is an additive continuation.

HalfCheetah and the optional Gaussian/biased-channel experiment were not run.
