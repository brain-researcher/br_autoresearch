# EP19 scope, novelty and clarity review — 2026-10-02

Question: does past EEG improve 300–600 ms onset forecasts beyond measured context and peripheral history, and does the increment and its input-block dependence repeat under a second-task procedure?

## Closest work and the remaining contribution

[Prediction of human voluntary movement before it occurs](https://pmc.ncbi.nlm.nih.gov/articles/PMC5558611/) already demonstrates premovement EEG prediction. [Crell et al. (2025)](https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2025.1540155/full) studies transfer from cued training to asynchronous self-paced detection and reports false-alarm operating points. The [WAY source](https://pmc.ncbi.nlm.nih.gov/articles/PMC4365902/) and [self-paced dataset](https://www.nature.com/articles/s41597-025-06039-9) supply the two task settings, not independent evidence for EP19's proposed result.

Lead time, slow EEG, asynchronous detection and cross-task evaluation are not individually novel. EP19's defensible target is continuous natural-risk forecasting conditional on available sensor history, matched surrogate comparisons and held-out participant repetition of the registered recent-versus-older-slow input-support prediction.

## Decision and repair

Keep the core design. Explicit prior-work contrasts and positive/null boundaries replace a generic gap statement. Model-family ordering and new architecture search remain secondary; they should not turn the study into several unrelated papers.

A positive increment is conditional on the measured sensors and their onset definition, not proof of cortical origin or prediction before every peripheral change. A null means that the tested models did not establish the specified incremental forecast; it does not establish absence of motor information in EEG. Input ablations identify model dependence, not distinct physiological generators. The second source repeats a procedure with its own development-trained encoder; it is not zero-shot transfer of WAY weights.

The paper remains scientifically plausible but crowded and conditional on consequential replicated increments or a well-resolved boundary. No claim of first premovement decoding is warranted.

No horizon, model panel, margin, calibration budget, search scope or access permission changes. No empirical data were scored.
