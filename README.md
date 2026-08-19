# Brain Researcher Autoresearch

We have Brain Researcher as an analytical tool. The next step is to see whether
this kind of agentic workflow can be extended into autonomous, self-evolving
research episodes.

We plan to run more than 100 neuroimaging research episodes. We want to use
them to do two things:

1. accelerate neuroimaging discovery; and
2. ask whether we are truly accelerating discovery, or simply automating
   p-hacking and analysis selection.

We assume that an autoresearch episode can be launched, can run, and can reach
a clear ending. We do not assume that it will produce a positive scientific
result. A good episode may find a candidate, rule out an idea, finish with no
candidate, or identify a technical limitation. All of these outcomes should be
kept.

The research question can be narrow or very open. We deliberately want a large
search space: different explanations, estimands, representations, models,
controls, validation strategies, and datasets may all be considered. One aim
of the campaign is to see how broadly an agent can search, how deeply it can
follow the most promising branches, and whether that produces more informative
and reproducible science.

One episode does not pretend to exhaust that space. It first maps several
meaningfully different candidate directions, then carries at most one or two
into deeper falsification, ablation, sensitivity, and negative-control work.
Unexplored branches and failed ideas remain available to later episodes. Across
more than 100 episodes, we can study both breadth and depth without confusing a
larger search with a license to select whichever analysis happens to look best.
Here, a better result means a sharper test, a more reproducible or transportable
finding, or a well-supported reason to stop. It does not mean a smaller
`p`-value.

## How the loop is designed

The basic loop is:

```text
question + data
    -> explore competing explanations and tests
    -> Society critique
    -> scientist gate
    -> approved experiment
    -> finding, refutation, or limitation
    -> next question
```

### 1. Start with a question

Each episode starts with `GOAL.md` and `DATASETS.md`.

The Goal can be specific:

> Does a category-balanced 2-back versus 0-back pattern remain reliable across
> sessions after controlling for motion and task difficulty?

It can also be broad:

> I want to know which parts of the n-back response are stable across sessions
> and cohorts, and which parts only work for one task or site.

Brain Researcher can help turn either form into a testable research question.
The scientist still decides whether the rewritten Goal is the right question.

### 2. Generate hypotheses before searching for a result

The agent first looks at the phenomenon, the available data, and what is
already known. It then proposes competing explanations, not just different
ways to obtain a significant result.

For each candidate it should state:

- the proposed explanation;
- what observation would support or weaken it;
- a falsifier;
- the main confounds and negative controls;
- whether the required data are actually available; and
- the approximate compute and data cost.

Under the ordinary bounded-search policy, the runner can generate two to eight
candidates and carry forward at most two that are ready to test. The candidate
set and the comparison plan are recorded before the agent reads the outcomes
that would decide among them.

The agent may use the following as idea sources:

- the current [Research Landscape](https://brain-researcher.com/tbsci-9d4f41e7358c#landscape);
- the current [Self-evolvable Questions](https://brain-researcher.com/tbsci-9d4f41e7358c#questions);
- findings and unresolved questions from earlier episodes;
- Brain Researcher's 15 research routes; and
- optional OOD hypothesis generation from the Brain Researcher knowledge
  graph.

Landscape and Questions entries are useful seeds, not required inputs. OOD
generation looks for an unusual connection in the knowledge graph and turns it
into a hypothesis, a small discriminating test, a falsifier, and controls. An
OOD idea is still only a candidate, not a finding.

### 3. Let Society critique the candidate

If discovery produces a candidate worth carrying forward, the candidate and
its evidence are frozen and sent to Brain Researcher Society, a multi-agent
scientific review system that separates independent critique, adversarial
challenge, evidence integration, and final human authority.

```text
frozen candidate
    -> independent reviews
    -> cross-check and red team
    -> advisory synthesis
    -> strict eligibility gate
    -> scientist reward and launch decision
```

For Goal-candidate review, the current implementation uses ten reviewing
agents coordinated by a root conductor. That is an internal review instrument,
not ten extra steps that every episode must expose to its user. The panel can
recommend; it cannot reward a direction, authorize an experiment, accept a
claim, or update the Landscape. Those powers remain separated.

If an episode ends with no candidate or with a technical failure, it stops
without Society review.

The current topology, implementation boundaries, information design, power
separation, and plans for cross-episode reviewer calibration are described in
[SOCIETY.md](SOCIETY.md). Whether the full ten-agent panel is better than a
smaller review is an empirical question for this campaign, not an assumption.

### 4. Run the experiment and turn the result into findings

Only a scientist-approved experiment is run as confirmation. Finishing the
computation does not automatically make the hypothesis true.

For a registered confirmation episode that freezes the scientific-learning
closeout policy, the server reviews the terminal result and can produce a
scientific review and report. Other runs stop at their terminal facts until a
separate review is requested. When the scientific-learning path is used, the
episode records:

- what the experiment actually tested;
- what was supported, refuted, or still unresolved;
- the strongest remaining alternative explanation;
- where the result is expected to generalize and where it may fail;
- failures, deviations, and resource use; and
- the next question suggested by the evidence.

The closeout is still non-promoting: it does not automatically create a
ClaimCard or change the Landscape. Reviewed outputs can motivate a successor
episode with a different test. Any shared finding, Questions update, or
Landscape transition still needs its own persisted review and transition
authority.

This is what we mean by **self-evolving**: the evidence from one episode changes
which questions and experiments are proposed next. It does not mean that an
agent can approve its own claim or silently launch the next experiment.

## The 15 Brain Researcher research routes

The routes are general ways to reformulate a problem. They are not 15 fixed
analysis pipelines.

1. audit an assumption and pivot if it is wrong;
2. replace an operator, model component, or representation;
3. redesign a fixed part of the data-generating process;
4. design a controlled comparison that separates competing explanations;
5. put different datasets or measurements into a shared representation;
6. rewrite the question as a more solvable object;
7. create a self-supervised learning signal from the data;
8. build known scientific structure into the method;
9. connect methods that are algebraically equivalent;
10. split heterogeneous data into parts that need different treatment;
11. split a large problem into smaller problems and delegate them;
12. turn a discrete search into a continuous one;
13. adapt a model by conditioning instead of retraining it;
14. measure a limit first, then try to exceed it; and
15. design a self-supervised objective for the property we want to measure.

An episode may use one, several, or none of these routes.

## Data already available on Sherlock

The first version of the campaign will mainly explore data that have already
been processed on Sherlock.

As of 2026-08-19, the main shared starting points are:

| Data | Count | Path |
| --- | ---: | --- |
| OpenNeuro FitLins result directories | 54 | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses` |
| OpenNeuro fMRIPrep directories | 9 | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep` |
| HCP-YA derivatives | not yet inventoried here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP-YA-BIDS` |
| HCP connectivity data | not yet inventoried here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP1200_PTN` |

These counts mean that the directories exist. Each episode still checks the
task, subjects, events, contrasts, confounds, files, access rules, and which
data must remain untouched for confirmation.

## What someone needs to submit

The main inputs are two files:

```text
episodeNN_short_name/
├── GOAL.md
├── DATASETS.md
├── inputs/README.md
└── outputs/README.md
```

`GOAL.md` says what you want to know. It can be specific, broad, or include
possible analysis methods. Brain Researcher can help rephrase it.

`DATASETS.md` says which data you want to use, where they are, what is already
known, and what still needs to be checked. Imaging data do not go into Git.

Once the Goal looks right, open a pull request. I will review the proposal,
merge accepted episodes, and run them together on Sherlock. I currently have
the model-token budget to support these runs.

If you have another dataset you want to explore, tell me what it is and what
question you want to ask. If its access rules allow it, I can help put it on
Sherlock.

## Repository record

Every episode is tracked from the time its Goal is proposed. We keep its Goal,
dataset notes, code, analysis choices, failed attempts, figures, tables,
reports, and final status. Large imaging data and temporary compute files stay
outside Git.

- [CAMPAIGN.md](CAMPAIGN.md) lists the episodes and their current status.
- [LANDSCAPE.md](LANDSCAPE.md) summarizes the research areas we are exploring.
- [QUESTIONS.md](QUESTIONS.md) contains possible questions for future episodes.
- [SOCIETY.md](SOCIETY.md) explains the multi-agent review and its limits.
- [AGENTS.md](AGENTS.md) contains the instructions used by agents running here.
