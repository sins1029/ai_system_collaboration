# SAC warm-start comparison

Three paired seeds run 10,000 real SustainCluster environment interactions per group. Actor architecture, critic initialization, replay capacity, reward, optimizer settings and evaluation seeds are identical; only actor initialization differs.

Reward threshold: -2731.328736

| Group | Initial reward | Final reward | Initial SLA | Final SLA | First 10% reward | First 25% reward | Steps to threshold | Expert agreement initial/final |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_init_sac | -3641.4356 +/- 765.5887 | -2359.9259 +/- 102.2302 | 966.00 | 1977.00 | -30.498673 | -29.109465 | 666.6666666666666 | 0.7796/0.7668 |
| bc_init_sac | -1821.2219 +/- 0.0000 | -2432.2136 +/- 0.0000 | 936.00 | 2080.00 | -31.537318 | -32.320689 | 0 | 0.9256/0.7690 |

The original one-million-transition replay setting is not used because `1,000,000 x 750 x 233 x 2` float observations would require roughly 1.4 TB before actions and masks. Both groups instead use the same bounded replay configuration recorded in JSON.

A fall in expert agreement after updates is treated as policy drift; reward/SLA changes determine whether that drift is useful adaptation or behavior-cloning distribution shift.

## Answers

1. **Initial performance:** yes. BC improves mean initial evaluation reward by 1820.2137 (higher is better).
2. **Early SLA:** yes at initialization. BC reduces initial SLA violations by 30.00 per evaluation episode, although this advantage does not survive fine-tuning.
3. **Reward threshold:** BC reaches the fixed threshold at step 0; random initialization needs 666.6666666666666 steps on average among the paired runs.
4. **Seed variance:** yes initially. BC initial reward std is 0.000000, versus 765.588733 for random initialization.
5. **MPC policy drift:** yes. BC expert agreement changes by -0.156677, and evaluation reward changes by -610.9917 after 10,000 steps.
6. **Distribution shift:** yes. The BC actor starts close to the expert, but online SAC updates reduce expert agreement and increase final SLA violations; the current reward/update formulation does not preserve the cloned safety behavior.

This is a warm-start validation, not evidence that the current SAC fine-tuning recipe is ready for long training. A KL/BC regularizer, expert replay mixing, or a frozen-actor critic warmup should be evaluated before scaling.
