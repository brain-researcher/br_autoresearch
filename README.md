# Brain Researcher Autoresearch

Browse the [EP01-EP21 figure gallery](FIGURES.md) for current study schematics
and links to each episode's scientific goal. Scope, prior work, and claim limits
are maintained in the episode documents, not separate review reports.

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

The episode's scientific policy sets the breadth, depth, and resource limits
of its search. An ordinary bounded-search lane maps competing directions and
selects a small set for deeper falsification. The
[adaptive-search protocol](ADAPTIVE_SEARCH_PROTOCOL.md) supports many more
outcome-adaptive successor cycles inside a frozen grammar, append-only ledger,
finite round budget, and one-shot audit. Across more than 100 episodes, we can
study both breadth and depth without confusing a larger search with a license
to select whichever analysis happens to look best. Here, a better result means
a sharper test, a more reproducible or transportable finding, or a
well-supported reason to stop. It does not mean a smaller `p`-value.

## Which skill to use

| What you want to do | Skill |
| --- | --- |
| Turn a question or dataset into an EP; assess novelty; revise its design, writing or concept figure | [`design-episode`](.agents/skills/design-episode/SKILL.md) |
| Implement or run an existing EP; repair, monitor, analyze and show its results | [`run-episode`](.agents/skills/run-episode/SKILL.md) |
| Explicitly manage formal BR campaign state, Society review, reward or registered confirmation | [`manage-research-campaign`](.agents/skills/manage-research-campaign/SKILL.md) |

The first two are the everyday research workflows. The third is optional,
not a prerequisite or the next step after every experiment. Designing an EP
does not launch it, and requesting a status report does not authorize a repair
or another run. Scientific critique can inform either everyday workflow without
starting a formal campaign.

The former `br-autoresearch-episode` is now `run-episode`;
`brain-autoresearch-loop` is now `manage-research-campaign`.
`design-episode` adds the previously missing study-authoring entrypoint.
Old names are not duplicate discoverable skills. Existing episode contracts,
run identities and frozen campaign protocols are unchanged by this rename.

## How the loops are designed

Standalone episode work is the default: an explicit scientist request starts
work under the named episode's contract. Local exploration and evaluation do
not require Society review or canonical registration. Formal Society review,
reward, confirmation, and shared-claim promotion are a separate workflow used
when explicitly requested, with authority checked before the relevant action.

For an episode adopting the adaptive-search protocol, development proceeds
inside one authorized round:

```text
authorized round + frozen policy
    -> branch coverage
    -> hypothesis -> development trial -> falsifier -> successor
    -> incumbent/challenger decisions until a valid stop
    -> configuration lock
    -> episode-managed locked held-out evaluation
    -> finding, refutation, limitation, or incomplete search
```

Development trials remain local evidence. They do not individually become
accepted findings or authorize a new scientific round.

### 1. Start with a question

An idea can start with a question, a dataset, or a permitted prior observation.
Use [design-episode](.agents/skills/design-episode/SKILL.md) to establish the
closest prior work, useful scientific distinction, data-supported comparison,
and aligned study narrative/figure. It keeps exploratory choices open without
inventing universal pilot sizes or candidate caps.

An executable episode has `GOAL.md`, `DATASETS.md`, and one search policy,
normally `SEARCH_POLICY.yaml` (EP02 uses JSON). Use
[run-episode](.agents/skills/run-episode/SKILL.md) once execution is requested.
[AGENTS.md](AGENTS.md) defines workspace and data boundaries, targeted checks,
execution records, and canonical authority.
Scientific payloads remain read-only; source documentation can be maintained
within those boundaries. Checks and held-out evaluation occur when required
by the active scientific stage.

The Goal can be specific:

> Do condition-description features reduce held-out-dataset map error beyond a
> task-family prototype under a dataset-grouped split?

It can also be broad:

> I want to know which aspects of an experimental condition predict brain-map
> geometry across studies, and where that relationship stops generalizing.

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

When an episode adopts the ordinary bounded-search policy, the runner can
generate two to eight candidates and carry forward at most two that are ready
to test. That candidate set and comparison plan are recorded before outcomes
that decide among them are read. These numerical limits do not apply to the
adaptive-search lane unless its episode policy adopts them; that lane permits
outcome-adaptive successors inside its frozen grammar and budget.

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

### 3. Request canonical review when needed

When the scientist requests formal Society review, the frozen candidate and
its evidence enter Brain Researcher's review workflow. Independent critique,
adversarial challenge, and synthesis inform the scientist's decision. The
panel cannot reward a direction, authorize confirmation, accept a claim, or
update the Landscape itself. A local candidate does not automatically start
this workflow.

[SOCIETY.md](SOCIETY.md) describes the review design and its limits;
[AGENTS.md](AGENTS.md) defines when to use canonical operations. Whether a
larger panel improves scientific review remains an empirical campaign question.

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

As of 2026-09-23, the main shared starting points are:

| Data | Count | Path |
| --- | ---: | --- |
| OpenNeuro FitLins `ds*` result directories | 55 | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/analyses` |
| OpenNeuro fMRIPrep directories | 9 | `/oak/stanford/groups/russpold/data/OpenNeuro_analyses/openneuro_fitlins/fmriprep` |
| HCP-YA derivatives | not yet inventoried here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP-YA-BIDS` |
| HCP connectivity data | not yet inventoried here | `/oak/stanford/groups/russpold/data/HCP_YA/HCP1200_PTN` |

These counts mean that the directories exist. Each episode still checks the
task, subjects, events, contrasts, confounds, files, access rules, and which
data must remain untouched for confirmation.

See [DATA_CATALOG.md](DATA_CATALOG.md) for the shared releases, versions,
storage locations, and access notes currently available on Sherlock.

## What someone needs to submit

The episode contract and initial workspace are:

```text
episodeNN_short_name/
|-- GOAL.md
|-- DATASETS.md
|-- SEARCH_POLICY.yaml (or SEARCH_POLICY.json)
|-- inputs/README.md
`-- outputs/README.md
```

`GOAL.md` says what you want to know. It can be specific, broad, or include
possible analysis methods. Brain Researcher can help rephrase it.

`DATASETS.md` says which data you want to use, where they are, what is already
known, and what still needs to be checked. Imaging data do not go into Git.

The search policy freezes the admissible grammar, objectives, branch coverage,
budgets, falsifiers, stopping rule, and one-shot audit boundary.

## ASTRA experiment program

Maintain `outputs/astra/v0.0.14/astra.yaml` as the episode's full experiment
program: one root analysis with an `analyses` entry for each independently
runnable or evaluable experiment already required by its contracts. Inputs
and outputs express dependencies. Attempts, job IDs, failures, artifacts, and
next actions belong in `outputs/experiment_log.md`, keyed by analysis ID.
Follow the [ASTRA authoring reference](.agents/skills/run-episode/references/astra.md)
through [run-episode](.agents/skills/run-episode/SKILL.md).

The legacy `bin/astra-milestone` adapter emits a lossy single-analysis plan or
terminal projection. It is not a replacement for the full program and must
not overwrite it. ASTRA records do not approve execution, accept claims,
record reward, or update the Landscape. An authoring or validation failure
leaves ASTRA compliance open without blocking otherwise-authorized work.

To propose a new episode for campaign inclusion, submit its contract for
review through a pull request when that Git work is requested. Proposal review
is separate from starting already-authorized standalone work.

If you have another dataset you want to explore, tell me what it is and what
question you want to ask. If its access rules allow it, I can help put it on
Sherlock.

## Repository record

Each formal episode has one current Goal, dataset contract, and search policy.
Once candidate-scoring adaptive execution starts, its scored trials,
challenger decisions, and execution failures belong in the ordered trial log
and current output workspace; acquisition and preliminary readiness need only
the lean decision record required by the scientific question. Preserve frozen
paths, active writers, and protected history. Mark obsolete navigation as
historical; moving or deleting artifacts requires a separately scoped change.
Large imaging data and temporary compute files stay outside Git.

- [CAMPAIGN.md](CAMPAIGN.md) lists formal episode slots, their current status,
  immutable prior runs, and reserved IDs.
- [DATA_CATALOG.md](DATA_CATALOG.md) lists shared datasets available on
  Sherlock; episode-specific eligibility and audit boundaries remain governed
  by each episode's dataset contract.
- [EPISODE_REGISTRY.yaml](EPISODE_REGISTRY.yaml) separates stable episode IDs,
  historical prior identities, direct physical paths, and current policy
  references. The current EP01 is `episode01_narps_deep_search`; the old NARPS
  runs are frozen in the [historical prior archive](_examples/historical_prior_records/).
- [Adaptive-search protocol](ADAPTIVE_SEARCH_PROTOCOL.md) and
  [controller interface](ADAPTIVE_CONTROLLER_INTERFACE.md) define the search,
  append-only lineage, and one-shot audit boundary. A design becomes realized
  evidence only when its episode records the corresponding run artifacts.
- [Unnumbered examples](_examples/README.md) retain reusable design provenance;
  the NARPS deep-search example is retired and is not the current EP01
  scientific contract.
- [LANDSCAPE.md](LANDSCAPE.md) summarizes the research areas we are exploring.
- [QUESTIONS.md](QUESTIONS.md) contains possible questions for future episodes.
- [Foundation-20 crosswalk](portfolios/foundation20/CROSSWALK.md) records direct,
  bounded, partial, adjacent, and missing topic coverage; its exposure ledger
  prevents shared data from being mislabeled as independent confirmation.
- [SOCIETY.md](SOCIETY.md) explains the multi-agent review and its limits.
- [AGENTS.md](AGENTS.md) contains the instructions used by agents running here.
