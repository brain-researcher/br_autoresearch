# EP23 literature verification (2026-10-13)

Produced by Codex `gpt-6-astra` (`max`, read-only, web search; no local files or data) from the design draft's claim list, then reviewed by the orchestrator. Spot checks against primary pages: Nava & Wyart (arXiv 2605.23821 abstract), Kang, Canatar & Chung (arXiv 2502.19648 abstract; the venue is not stated on that page), and Khaligh-Razavi & Kriegeskorte 2014 (PLOS Computational Biology article). L-numbers refer to the claim list in the design brief. "UNVERIFIED" marks items the audit could not resolve. GOAL.md summarizes the consequences.

Audit date: **12 October 2026**. “Yes” below means the draft’s **qualified summary** is supported; it does not endorse a universal convergence theorem. **UNVERIFIED** identifies checks that retrieval or the supplied material could not resolve.

| ID | Verified citation and inspected version | What it actually shows, ≤2 sentences | Stated conditions / assumptions | Draft usage accurate? | Exact correction if not “yes” |
|---|---|---|---|---|---|
| **L01** | Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola. **“Position: The Platonic Representation Hypothesis.”** ICML 2024. [Proceedings](https://proceedings.mlr.press/v235/huh24a.html); [arXiv:2405.07987v3](https://arxiv.org/html/2405.07987v3). | Population-optimal contrastive scores recover a co-occurrence PMI density ratio, with objective-dependent offsets. Bijective observations allow the same latent co-occurrence structure to be expressed across modalities. | Finite idealized world; deterministic bijective observation maps; positives from the specified co-occurrence distribution; independent marginal negatives; population optimization and realizable scores. Binary NCE gives PMI plus log prior odds; InfoNCE permits an anchor-dependent additive constant, reduced to a common constant for a realizable symmetric score. A PMI matrix is **not automatically PSD**: Appendix F supplies sufficient conditions for a constant-shifted Gram representation. No general finite-data or optimizer-convergence theorem. | **Yes** | — |
| **L02** | Fabian Gröger, Shuo Wen, Maria Brbić. **“Revisiting the Platonic Representation Hypothesis: An Aristotelian View.”** ICML 2026, PMLR 306. [Proceedings](https://proceedings.mlr.press/v306/groger26a.html); [arXiv:2602.14486v2](https://arxiv.org/html/2602.14486v2). | Calibration substantially weakens capacity trends in the tested global metrics while retaining neighbourhood alignment. Image–text and video–text experiments support this qualified distinction. | Null: permute sample correspondence in one representation, preserve its marginal geometry, and repeat layer-selection/aggregation under each permutation. Global measures include CKA/SVCCA/Procrustes; mKNN measures shared neighbour identities; small-bandwidth RBF-CKA probes local metric agreement. Coverage includes WIT, PVD, language-model families, VideoMAE, and frame-based DINOv2/CLIP. Valid permutation inference requires exchangeability or appropriate restricted permutations. | **Partly** | Replace “global convergence largely disappears” with **“capacity-dependent convergence in the tested global spectral/distance measures largely disappears after aggregation-aware null calibration.”** This does not exclude global relational structure; L03 directly demonstrates that distinction. |
| **L03** | Junwon You, Mihyun Jang, Sangwoo Mo, Jae-Hun Jung. **“What Converges in the Platonic Representation Hypothesis? Structure over Geometry.”** [arXiv:2609.27252v1](https://arxiv.org/html/2609.27252v1), 23 September 2026, preprint. | Relational alignment remains stronger than increasingly stringent distance agreement at both tested scales. The pattern survives permutation calibration, family comparisons, normalization checks, and a Riemannian-distance approximation. | Main experiment: 1,024 WIT pairs, 12 language models and 17 vision transformers; extension includes VideoMAE and frame-based DINOv2/CLIP against text. **Local** means kNN edges; **global** means the \(H_0\) death-edge/MST spanning skeleton, with deterministic tie handling. Distance-aware variants weight shared edges by normalized distance agreement. These scales are **not equivalent to between-category versus within-category comparisons**. | **Yes** | — |
| **L04** | Jeff Z. HaoChen, Colin Wei, Adrien Gaidon, Tengyu Ma. **“Provable Guarantees for Self-Supervised Deep Learning with Spectral Contrastive Loss.”** NeurIPS 2021; inspected [arXiv:2106.04156v7](https://arxiv.org/pdf/2106.04156v7), 2022 revision. | A specific quadratic contrastive loss becomes low-rank approximation of a normalized positive-pair graph. Separate theorems bound downstream linear-probe error and finite-sample generalization. | \(A_{xx'}=\mathbb E_{\bar x}[p(x\mid\bar x)p(x'\mid\bar x)]\), \(d_x=\sum_{x'}A_{xx'}\), \(\bar A=D^{-1/2}AD^{-1/2}\). Loss: \(-2\mathbb E_+\langle f,f'\rangle+\mathbb E_{\rm independent}\langle f,f'\rangle^2\). With \(F_x=\sqrt{d_x}f(x)\), unrestricted global optima factor the leading spectrum of \(\bar A\). Downstream results require label recoverability, graph expansion and realizability; population dimension \(k\ge2r\), with stronger finite-sample requirements including \(k\ge\max(4r+2,2m)\). Finite-sample bounds include complexity and spectral-gap terms; they do not guarantee SGD finds the optimum. | **Yes** | — |
| **L05** | Omer Levy, Yoav Goldberg. **“Neural Word Embedding as Implicit Matrix Factorization.”** NeurIPS 2014. [Published paper](https://papers.neurips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf). | Unconstrained SGNS word–context scores optimize a shifted log co-occurrence ratio. Low-dimensional SGNS performs a weighted logistic factorization, rather than exact unweighted SVD. | Separate word/context factors \(W,C\). Per-cell optimum is \(\log[p(w,c)/(p(w)q(c))]-\log k\); ordinary shifted PMI requires \(q(c)=p(c)\), while \(q\propto p(c)^{3/4}\) changes the target. Positive and negative counts determine weighting; limited rank couples cellwise optima. A symmetric PSD kernel additionally requires compatible word/context indexing, symmetric statistics, appropriate negatives, **PSD of the resulting target**, and tied/otherwise identified factors with sufficient dimension. Symmetry alone does not imply PSD; \(WC^\top\) does not identify \(WW^\top\). | **Yes** | — |
| **L06** | Zhiquan Tan, Yifan Zhang, Jingqin Yang, Yang Yuan. **“Contrastive Learning is Spectral Clustering on Similarity Graph.”** ICLR 2024; [arXiv:2303.15103v4](https://arxiv.org/html/2303.15103v4). | InfoNCE is rewritten as graph attraction plus an embedding-dependent repulsion/partition term. “Spectral clustering” uses the paper’s regularized graph-embedding definition, not simply an eigenvector solution. | Finite full object set; augmentation probabilities define \(\pi_{ij}\); SimCLR theorem assumes symmetric sampling. Unit-normalized dot products at temperature \(\tau>0\) correspond to Gaussian affinities \(\exp[-\lVert z_i-z_j\rVert^2/(2\tau)]\). The objective retains \(\log R(Z)\) alongside a Laplacian quadratic form; network realizability and normalization constrain \(Z\). The finite full-set rewrite requires no infinite-negative limit, but ordinary minibatch training differs from this full-set objective. | **Yes** | — |
| **L07** | Benjamin Eysenbach, Tianjun Zhang, Ruslan Salakhutdinov, Sergey Levine. **“Contrastive Learning as Goal-Conditioned Reinforcement Learning.”** NeurIPS 2022. [arXiv:2206.07568](https://arxiv.org/html/2206.07568). | The population-optimal binary contrastive critic is a log discounted-occupancy/marginal ratio, related to a particular goal-reaching Q-function. It is not established as a spectral function of a transition operator. | \(d^\pi=(1-\gamma)\sum_{t\ge0}\gamma^t p_t^\pi\), \(0\le\gamma<1\). Positives are discounted future states; negatives follow their replay-induced marginal \(p(s_f)\). Balanced binary NCE gives \(f^*(s,a,s_f)=\log[d^{\bar\pi}(s_f\mid s,a)/p(s_f)]\). Here occupancy averages over commanded goals conditional on replay state/action; it is not automatically the policy conditioned on the queried goal. Realizable population optimum is required. Entrywise logarithms and marginal division do not preserve the transition eigenbasis in general. | **Yes** | — |
| **L08** | Andres Nava, Matthieu Wyart. **“Hierarchical Concept Geometry in Language Models Emerges from Word Co-occurrence.”** [arXiv:2605.23821v1](https://arxiv.org/html/2605.23821v1), May 2026, preprint. | An idealized tree-distance co-occurrence kernel has hierarchy-adapted eigenspaces and ordered leading split modes. Word2vec and Gemma 2B **unembedding** experiments test corresponding WordNet geometry. | Full regular binary-tree setting; noise-free \(M_{ij}=f(\mathrm{dist}(i,j))\); positivity gives the leading positive scaling mode, strict decrease gives sign-coherent leading split modes. Block interlacing orders leading split eigenvalues; it does not completely order every mode. The normalized co-occurrence statistic only approximates PMI near independence. Results primarily concern spectral splitting and induced geometry, not universal distance/kNN recovery. | **Yes** | — |
| **L09** | **(a)** Geoffrey Roeder, Luke Metz, Kingma. **“On Linear Identifiability of Learned Representations.”** [ICML 2021](https://proceedings.mlr.press/v139/roeder21a.html).<br>**(b)** Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, Wieland Brendel. **“Contrastive Learning Inverts the Data Generating Process.”** [ICML 2021](https://proceedings.mlr.press/v139/zimmermann21a.html).<br>**(c)** Imant Daunhawer, Alice Bizeul, Emanuele Palumbo, Alexander Marx, Julia E. Vogt. **“Identifiability Results for Multimodal Contrastive Learning.”** ICLR 2023; [arXiv:2303.09166](https://arxiv.org/html/2303.09166).<br>**(d)** Dingling Yao, Danru Xu, Sébastien Lachapelle, Sara Magliacane, Perouz Taslakian, Georg Martius, Julius von Kügelgen, Francesco Locatello. **“Multi-View Causal Representation Learning with Partial Observability.”** ICLR 2024; [arXiv:2311.04056](https://arxiv.org/html/2311.04056).<br>**(e)** Beatrix M. G. Nielsen, Emanuele Marconato, Andrea Dittadi, Luigi Gresele. **“When Does Closeness in Distribution Imply Representational Similarity? An Identifiability Perspective.”** NeurIPS 2025; [arXiv:2506.03784v2](https://arxiv.org/html/2506.03784v2), [proceedings PDF](https://proceedings.neurips.cc/paper_files/paper/2025/file/93fce71def4e3cf418918805455d436f-Paper-Conference.pdf). | Four conditional identification results and one distributional-proximity counterexample establish meaningful boundaries. None is a general theorem about convergence of arbitrary trained models. | **(a)** Bilinear conditional exponential-family model, equal modeled distributions, and full-rank/diversity conditions imply invertible linear equivalence. **(b)** Differentiable injective generator, uniform hyperspherical latents, vMF positive pairs, matched similarity family, population minimization and infinitely many negatives imply orthogonal identification, with the specified scale. **(c)** Smooth invertible modality maps, shared content, sufficiently varying styles, correct content dimension, and an asymptotic alignment/entropy optimum identify content up to an invertible map. **(d)** Paired partially overlapping views, smooth invertible observation maps and suitable latent-density/overlap assumptions identify shared blocks; causal dependence is allowed, but general causal-graph recovery does not follow. **(e)** In a bilinear-softmax family with dimension \(M\ge2\) and enough labels, increasing unembedding norms makes conditional KL vanish while fixed embeddings retain positive best-linear-matching error. This is a counterexample to **KL-based** guarantees, not every possible distributional distance. | **Yes** | — |
| **L10** | Katherine R. Storrs, Tim C. Kietzmann, Alexander Walther, Johannes Mehrer, Nikolaus Kriegeskorte. **“Diverse Deep Neural Networks All Predict Human Inferior Temporal Cortex Well, After Training and Fitting.”** *J. Cognitive Neuroscience* 33, 2044–2064, 2021. [DOI](https://doi.org/10.1162/jocn_a_01755); [author PDF](https://kriegeskortelab.zuckermaninstitute.columbia.edu/sites/default/files/content/StorrsKriegeskorte_2021_JOCN.pdf). | Nine trained CNN architectures achieved similarly high fitted prediction of human IT geometry in the tested experiment. The abstract reports 57% more explainable variance after training and a further 124% fitting improvement. | Sixty-two images; human fMRI IT and V1; ImageNet-trained versus untrained networks. Fitting reweights principal-component features within layers and subsequently combines layers. Evaluation holds out **both images and subjects**. Findings concern this model set, stimulus set and fitted endpoint. | **Partly** | Replace “reweighting erases native-RSA differences” with **“cross-validated feature/layer fitting substantially reduced architecture-dependent differences, yielding equivalently high performance for the nine trained architectures in this experiment.”** |
| **L11** | Jörn Diedrichsen, Nikolaus Kriegeskorte. **“Representational models: A common framework for understanding encoding, pattern-component, and representational-similarity analysis.”** *PLOS Computational Biology* 13:e1005508, 2017. [Published article](https://doi.org/10.1371/journal.pcbi.1005508). | Encoding, PCM and RSA can express hypotheses about the same activity-pattern second moment. Feature priors and regularization determine which representational model is being tested. | For features \(F\) and weight covariance \(\Omega\succeq0\), the induced second moment is \(G=F\Omega F^\top\), up to scaling conventions. Isotropic Gaussian weight priors connect to ridge regularization; PCM integrates over Gaussian activity patterns. The treatment assumes Gaussian noise, independent partitions and appropriate condition/channel covariance handling, including spatial prewhitening. Arbitrary fitted metrics require their own estimation and validation scheme. | **Yes** | — |
| **L12** | Philipp Kaniuth, Martin N. Hebart. **“Feature-reweighted representational similarity analysis: A method for improving the fit between computational models, brains, and behavior.”** *NeuroImage* 257:119294, 2022. [DOI](https://doi.org/10.1016/j.neuroimage.2022.119294); [published PDF](https://pure.mpg.de/pubman/item/item_3384829_4/component/file_3451207/Kaniuth_2022.pdf). | Ridge regression combines feature-specific similarities/dissimilarities to predict a target RSM/RDM. Reweighting improves prediction and can alter model selection across neural and behavioural datasets. | The main method allows **signed coefficients**; nonnegative fitting is examined separately. Nested CV splits **stimuli/conditions**, not randomly selected RDM entries: outer fivefold repeated ten times, inner fivefold repeated five times. Fractional ridge selects regularization; held-out fit is evaluated using Pearson correlation. Featurewise weighting lacks the cross-feature terms of a general full metric. | **Yes** | —. For the proposed comparison, describe nonnegative diagonal PSD fitting as a constrained relative of FR-RSA; the published signed method is not automatically a PSD metric. |
| **L13** | Abdulkadir Canatar, Jenelle Feather, Albert Wakhloo, SueYeon Chung. **“A Spectral Theory of Neural Prediction and Alignment.”** NeurIPS 2023. [arXiv:2309.12821](https://arxiv.org/html/2309.12821); [proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/9308d1b7d4ae2d3e2e67ae94b1078bf7-Abstract-Conference.html). | Dataset-averaged ridge prediction depends jointly on kernel eigenvalues and neural-target power along their eigenvectors. Neither dimensionality nor an eigenspectrum alone determines predictivity. | \(E_g(p)=\sum_i W_iE_i(p)\), with \(W_i\) the normalized target energy along mode \(i\); \(E_i\) depends on sample size, eigenvalue and ridge parameter. The statistical-mechanics learning-curve calculation averages over sampled training sets. Empirical spectra approximate an underlying stimulus distribution; SI.1 explicitly notes resulting prediction error. Feature rescaling changes ridge geometry, but invertible rescaling does not necessarily change unregularized least-squares fitted values. | **Yes** | — |
| **L14** | Michael F. Bonner, Russell A. Epstein. **“Object representations in the human brain reflect the co-occurrence statistics of vision and language.”** *Nature Communications* 12:4081, 2021. [Published article](https://doi.org/10.1038/s41467-021-24368-2). | Object2vec most strongly predicts anterior PPA responses; language word2vec has stronger effects in object-selective LO/pFs. These are statistical encoding results, not identification of the brain’s learning algorithm. | Object2vec uses human object annotations from 22,210 ADE20K scenes, with scene membership defining context; final representation has eight PCs. Four participants viewed 81 categories with ten exemplars each. Ninefold validation holds out categories in separate runs. Controls include backgrounds chosen to reduce visual-feature correlations, AlexNet nuisance regressors, and object spatial properties. Subjects, categories and exemplars are different inferential units; 810 images do not imply 810 independent human observations. | **Yes** | — |
| **L15** | Kimberly L. Stachenfeld, Matthew M. Botvinick, Samuel J. Gershman. **“The hippocampus as a predictive map.”** *Nature Neuroscience* 20:1643–1653, 2017; corrected 2018 version inspected. [DOI](https://doi.org/10.1038/nn.4650); [author PDF](https://gershmanlab.com/pubs/Stachenfeld17.pdf). | A successor-representation model accounts for several place-field phenomena; SR eigenvectors produce grid-like and boundary-dependent patterns in simulations. The paper proposes a neural interpretation of this predictive basis. | Convention: \(M^\pi=\sum_{t\ge0}\gamma^t(P^\pi)^t=(I-\gamma P^\pi)^{-1}\), without occupancy normalization by \(1-\gamma\). Results depend on the behavioural transition policy and environment. Circular, square, trapezoidal, hairpin and compartmental environments produce different eigenvectors; many are irregular rather than canonical hexagonal grids. This is principally a computational account compared with existing recordings. | **Yes** | — |
| **L16** | **(a)** Anna C. Schapiro, Timothy T. Rogers, Natalia I. Cordova, Nicholas B. Turk-Browne, Matthew M. Botvinick. **“Neural representations of events arise from temporal community structure.”** *Nature Neuroscience* 16:486–492, 2013. [Author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC3749823/), DOI 10.1038/nn.3331.<br>**(b)** Mona M. Garvert, Raymond J. Dolan, Timothy E. J. Behrens. **“A map of abstract relational knowledge in the human hippocampal–entorhinal cortex.”** *eLife* 6:e17086, 2017. [Published article](https://elifesciences.org/articles/17086). | Schapiro links experimentally experienced communities to event segmentation and cortical similarity. Garvert finds hippocampal–entorhinal adaptation reflecting learned graph relationships, best captured by a successor-like associative measure. | **Schapiro:** 15-node, three-community graph with uniform available-edge transition probabilities; random-walk exposure and Hamiltonian-path controls; effects in left IFG, insula, ATL and STG. The hippocampal reanalysis is a later paper. **Garvert:** 23 participants; 12 object-labelled graph nodes with randomized identities, constrained random-walk exposure on day one, seven-object randomized transition testing on day two. The brain measurements are MVPA/adaptation readouts after controlled learning; neither establishes a universal spectral learning mechanism. | **Yes** | — |
| **L17** | Nuo Li, James J. DiCarlo. **“Unsupervised Natural Experience Rapidly Alters Invariant Object Representation in Visual Cortex.”** *Science* 321:1502–1507, 2008. [DOI](https://doi.org/10.1126/science.1160028); [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC3307055/). | Altering temporal contiguity changes adult macaque IT position tolerance in the predicted object- and position-specific direction. Significant single-neuron changes occur after about one hour; extended exposure can begin reversing preferences. | Two macaques; IT single-unit and multi-unit recordings; object identities swapped during saccades at a selected retinal position, with nonswapped/control comparisons and separate randomized response tests. The reported change is roughly five spikes/s per hour under targeted exposure. This tests plasticity of **position tolerance**, not arbitrary invariance, world-model learning, or cortical convergence. | **Yes** | — |
| **L18** | Dominik Schnaus, Thomas Dagès, Daniel Cremers, Xi Wang, Phillip Isola. **“Shared Geometry As A Rosetta Stone: Cross-Modal Alignment Without Paired Data.”** [arXiv:2610.09411v2](https://arxiv.org/html/2610.09411v2), 8 October 2026, preprint. | A geometrically initialized Wasserstein–Procrustes procedure estimates an orthogonal alignment from disjoint embedding sets. It succeeds on several modality pairs, including encoded NSD brain–brain alignment, but MEG–text is at chance: FOSCTTM approximately 0.498. | “Unpaired” concerns alignment fitting: retrieval evaluation uses true correspondences, and few-pair variants explicitly use pairs. NSD experiments reuse subject-specific encoders from L19, which use matched within-subject repetitions; results average over seeds rather than selecting the best seed. Success depends on compatible pretrained representation geometry. | **Partly** | Rosetta Stone is **broader than an NSD brain-alignment paper**. Write: “Rosetta Stone reports several unpaired embedding-alignment successes, including encoded NSD brain–brain alignment; its tested MEG–text alignment remains at chance, so unrestricted brain–other-modality alignment does not follow.” |
| **L19** | Pablo Marcos-Manchón, Rishi Jha, Lluís Fuentemilla. **“Platonic Representations in the Human Brain: Unsupervised Recovery of Universal Geometry.”** [arXiv:2605.20496v1](https://arxiv.org/html/2605.20496v1), May 2026, preprint. | Subject-specific NSD response encoders support strong orthogonal cross-subject alignment. The pipeline is unpaired across subjects during fitting, with important correspondence use elsewhere. | Encoders use matched repeated responses to the same image within each subject: reliability weighting, PCA/MCCA, linear distillation and an MLP contrastive encoder. Subject-unique images support fitting; 515 shared images support retrieval evaluation. Appendix C.2 selects the best of ten seeds using shared-image retrieval for the main results: mean rank 2.56 versus 4.61 for averaged rotations without seed selection. Rosetta Stone uses these encoders but reports seed averages. | **Partly** | State **“no cross-subject stimulus pairing for encoder/alignment fitting.”** Do not say “no correspondence information”: within-subject repeat correspondence is used, and the original headline retrieval result uses evaluation correspondence for seed selection. Its separate model–brain regression comparisons are paired. |
| **L20** | A. Emin Orhan, Brenden M. Lake. **“Learning high-level visual representations from a child’s perspective without strong inductive biases.”** *Nature Machine Intelligence* 6:271–283, 2024. [Published article](https://doi.org/10.1038/s42256-024-00802-0); [arXiv:2305.15372](https://arxiv.org/html/2305.15372). | SSL on child-headcam frames learns useful visual representations without category labels in pretraining. Public image-embedding checkpoints and loading code exist. | SAYCam: three children, approximately 472 hours total; frames sampled at 5 fps; individual-child and combined-data models differ. Objectives include DINO, Mugs and MAE, mainly ViTs, with additional architecture variants. Recording schedules, frame sampling, augmentations, architecture choices and labelled downstream evaluations introduce selection/human influence. [Code repository](https://github.com/eminorhan/silicon-menagerie) is MIT-licensed; [`dino_s_vitb14` weights](https://huggingface.co/eminorhan/dino_s_vitb14/tree/main) are listed, but a weight-specific license is **UNVERIFIED**. Static image encoding is documented; compatibility with EP23’s actual extraction pipeline is **UNVERIFIED**. | **Yes** | — |

For **L13**, the precise learning-curve quantities in SI.1 are

\[
E_g(p)=\frac{\kappa^2}{1-\gamma}
\sum_i\frac{W_i}{(p\lambda_i+\kappa)^2},
\qquad
\kappa=\alpha_{\rm reg}
+\kappa\sum_i\frac{\lambda_i}{p\lambda_i+\kappa},
\qquad
\gamma=\sum_i\frac{p\lambda_i^2}{(p\lambda_i+\kappa)^2}.
\]

Here \(p\) is training-set size, \(\lambda_i\) are eigenvalues of the kernel integral operator—empirically \(XX^\top/P\) under SI.1’s convention—and \(W_i\) is normalized neural-target energy along eigenfunction \(i\). This is a distribution-averaged prediction, not an identity for every finite train/test split. [Canatar et al., SI.1](https://arxiv.org/html/2309.12821)

**N01 — Novelty assessment: partly done; the exact integrated test was not found.**

Searches combined operator/eigenspace stability, objective-dependent spectral representations, neural measurement models, finite-neuron sampling, and within-category model–brain comparisons. Named searches covered every requested work. The closest overlaps are:

| Closest work | Overlap with the proposed contribution |
|---|---|
| Nava & Wyart, **“Hierarchical Concept Geometry…”**, 2026. [arXiv:2605.23821](https://arxiv.org/html/2605.23821) | Co-occurrence operators, hierarchical eigenspaces, coarse-to-fine splitting and perturbative robustness are already explicit. |
| Dhruva Karkada, Daniel J. Korchinski, Andres Nava, Matthieu Wyart, Yasaman Bahri. **“Symmetry in language statistics shapes the geometry of model representations.”** ICML 2026. [Proceedings](https://proceedings.mlr.press/v306/karkada26a.html), [paper](https://arxiv.org/html/2602.15029) | Statistical symmetries generate low-dimensional geometry; collective redundancy and eigenvalue separation support robustness, including stability of degenerate spectral pairs. |
| Randall Balestriero & Yann LeCun. **“Contrastive and Non-Contrastive Self-Supervised Learning Recover Global and Local Spectral Embedding Methods.”** NeurIPS 2022. [Proceedings](https://proceedings.nips.cc/paper_files/paper/2022/hash/aa56c74513a5e35768a11f4e82dd7ffb-Abstract-Conference.html) | Multiple SSL objectives already receive objective-specific spectral interpretations and optimal-representation analyses under their respective formulations. |
| Dan Yamins & Aran Nayebi. **“Contravariance Theory: Strong Alignment for Minimal Solutions to Hard Tasks.”** [arXiv:2607.08561v2](https://arxiv.org/html/2607.08561v2), 2026 | Gives conditional weak-to-strong and layerwise alignment results for constrained task solutions; directly overlaps attempts to explain representation convergence. |
| Nikolaus Kriegeskorte & Jörn Diedrichsen. **“Inferring brain-computational mechanisms with models of activity measurements.”** *Philosophical Transactions B* 371:20160278, 2016. [DOI](https://doi.org/10.1098/rstb.2016.0278), [author paper](https://arxiv.org/abs/1608.02027) | Explicit measurement models show that voxel averaging and sampling change apparent representational geometry; probabilistic RSA incorporates those effects. |
| Hyunmo Kang, Abdulkadir Canatar, SueYeon Chung. **“Spectral Analysis of Representational Similarity with Limited Neurons.”** NeurIPS 2025. [Paper](https://arxiv.org/html/2502.19648) | Especially close: neural sampling mixes/delocalizes eigendirections, biases CKA/CCA and can reverse model–brain rankings; theory assumes Gaussian linear projections, with power-law assumptions for its proposed spectral estimator. |
| Canatar, Feather, Wakhloo & Chung, **“A Spectral Theory of Neural Prediction and Alignment,”** NeurIPS 2023. [Paper](https://arxiv.org/html/2309.12821) | Already joins spectral weights, target alignment, stimulus distribution, sample size and regularized neural prediction. |
| Eric Elmoznino & Michael F. Bonner. **“High-performing neural network models of visual cortex benefit from high latent dimensionality.”** *PLOS Computational Biology*, 2024. [Published article](https://doi.org/10.1371/journal.pcbi.1011792) | Connects latent dimensionality with held-out neural encoding and generalization across models/tasks; dimensionality is not a novel explanatory variable here. |
| Laure Ciernik, Lorenz Linhardt, Marco Morik, Jonas Dippel, Simon Kornblith, Lukas Muttenthaler. **“Objective drives the consistency of representational similarity across datasets.”** ICML 2025. [Proceedings](https://proceedings.mlr.press/v267/ciernik25a.html) | Directly establishes objective-dependent dataset transfer of representational comparisons; the arXiv title begins “Training objective…”. |
| Zirui Chen & Michael F. Bonner. **“Universal dimensions of visual representation.”** [arXiv:2408.12804v2](https://arxiv.org/html/2408.12804v2), 2024 | Links dimensions shared across independently constructed vision networks to human fMRI alignment, overlapping independent model–model/model–brain predictions. |
| Seyed-Mahdi Khaligh-Razavi & Nikolaus Kriegeskorte. **“Deep Supervised, but Not Unsupervised, Models May Explain IT Cortical Representation.”** *PLOS Computational Biology*, 2014. [Published article](https://doi.org/10.1371/journal.pcbi.1003915) | Explicitly compares categorical divisions and within-category geometry, and distinguishes feature reweighting from general linear feature remixing. Thus both the scale contrast and diagonal-versus-full-readout idea have close precedents. |

Contravariance requires particularly careful wording. Its exact adjacent-layer theorem **assumes weak affine equivalence at both adjacent layers** and activation-specific “usedness.” Task hardness is formalized through the minimum required number of used nonlinear axes, \(m_\ell(\varepsilon)\); the alignment fraction is bounded below by \(m_\ell(\varepsilon)/d_\ell\). Full alignment requires all relevant axes to be used. The asymptotic result adds compactness and nondegeneracy conditions; the upstream “zippering” argument adds minimality and activation-specific identifiability. It does not establish that any two high-performing networks—or brains and networks—must align. [Contravariance, §§2–4 and appendices](https://arxiv.org/html/2607.08561v2)

The proposed contribution is therefore **partly done in substantial components**. I did **not find** a paper combining all four requested elements into the particular EP23 falsifiable test. That is a bounded search result, not evidence of firstness. The actual test specification was not supplied, so its distinctiveness remains **UNVERIFIED**.

A defensible replacement for N01 is:

> We combine existing objective-specific spectral results, perturbation bounds, and neural readout/measurement models into conditional propositions supporting a particular held-out structural test. The proposed contribution lies in the precise joint prediction and its empirical discrimination; priority for that combination has not been established.

**N02 — Applicable mathematical results and independent checks.**

The exact EP23 propositions, definitions of \(\varepsilon,\delta\), and proofs were not supplied. Their correctness is therefore **UNVERIFIED**. The following are precise statements suitable for checking or replacing them.

The standard references are:

- Carl Eckart & Gale Young, **“The Approximation of One Matrix by Another of Lower Rank,”** *Psychometrika* 1:211–218, 1936. [Primary journal record](https://doi.org/10.1007/BF02288367).
- Leon Mirsky, **“Symmetric Gauge Functions and Unitarily Invariant Norms,”** *Quarterly Journal of Mathematics* 11:50–59, 1960. [DOI](https://doi.org/10.1093/qmath/11.1.50).
- Chandler Davis & W. M. Kahan, **“The Rotation of Eigenvectors by a Perturbation. III,”** *SIAM Journal on Numerical Analysis* 7:1–46, 1970. [DOI](https://doi.org/10.1137/0707001).
- Yi Yu, Tengyao Wang & Richard J. Samworth, **“A useful variant of the Davis–Kahan theorem for statisticians,”** *Biometrika* 102:315–323, 2015. [DOI](https://doi.org/10.1093/biomet/asv008), [full published author PDF](https://personal.lse.ac.uk/wangt60/publication/DKvariant.pdf).

The original Eckart–Young and Mirsky full texts were not retrievable here; their bibliographic identities are verified. Original theorem numbering is **UNVERIFIED**. The approximation statements below also receive independent proofs. The perturbation constants were checked against the retrieved Yu–Wang–Samworth paper.

**Low-rank approximation.** Let \(A\in\mathbb R^{m\times n}\), with singular values
\(\sigma_1\ge\cdots\ge\sigma_q\ge0\), \(q=\min(m,n)\). For \(0\le r<q\), its truncated SVD \(A_r\) satisfies

\[
\min_{\operatorname{rank}(B)\le r}\|A-B\|_F
=\left(\sum_{i>r}\sigma_i^2\right)^{1/2},
\qquad
\min_{\operatorname{rank}(B)\le r}\|A-B\|_2
=\sigma_{r+1}.
\]

More generally, the same truncation minimizes every unitarily invariant norm. Frobenius minimization is unique when \(\sigma_r>\sigma_{r+1}\), or when \(r\ge\operatorname{rank}(A)\); a positive tie across the cutoff permits multiple minimizers. A singular-value gap does **not** generally make the operator-norm minimizer unique. [Eckart–Young](https://doi.org/10.1007/BF02288367), [Mirsky](https://doi.org/10.1093/qmath/11.1.50)

For \(K=U\operatorname{diag}(\lambda_i)U^\top\succeq0\), with descending eigenvalues,

\[
K_r=\sum_{i=1}^{r}\lambda_i u_i u_i^\top
\]

is simultaneously an unrestricted best rank-\(\le r\) approximation and a best **PSD** rank-\(\le r\) approximation. In particular,

\[
\min_{F\in\mathbb R^{n\times r}}\|K-FF^\top\|_F^2
=\sum_{i>r}\lambda_i^2,
\qquad
F=U_r\Lambda_r^{1/2}R,\quad R^\top R=I,
\]

with the usual zero padding and freedom inside tied eigenspaces.

For symmetric **indefinite** \(A\), the Frobenius-optimal PSD approximation instead keeps its largest positive eigenvalues:

\[
X_r^*=\sum_{i=1}^{r}\max(\lambda_i,0)\,u_i u_i^\top,
\qquad \lambda_1\ge\cdots\ge\lambda_n.
\]

It does not keep negative eigenvalues merely because their magnitudes are large.

*Independent proof.* For rank-\(\le r\) \(B\), the singular-value min–max principle gives
\(\sigma_j(A-B)\ge\sigma_{r+j}(A)\). Truncated SVD attains these lower bounds, establishing the norm claims. PSD truncation is feasible and attains the unrestricted bound. For indefinite symmetric \(A\), fixing candidate nonnegative eigenvalues \(x_i\), trace maximization aligns eigenvectors:

\[
\|A-X\|_F^2
=\|A\|_F^2+\|X\|_F^2-2\operatorname{tr}(AX)
\ge\sum_i(\lambda_i-x_i)^2.
\]

Minimizing with \(x_i\ge0\) and at most \(r\) nonzero values gives the stated positive-eigenvalue truncation.

These results require the specified **unweighted norm objective**. They do not directly solve weighted SGNS logistic loss, unit-diagonal Gram constraints, nonnegative-feature constraints, or arbitrary InfoNCE objectives.

**Symmetric subspace perturbation.** Let \(A,\widehat A=A+E\) be real symmetric, with eigenvalues ordered algebraically in descending order. Let \(U,\widehat U\) contain corresponding orthonormal eigenvectors indexed \(a,\ldots,b\), and let \(q=b-a+1\).

A classical Davis–Kahan form uses a **mixed** spectral separation: the interval \([\lambda_b,\lambda_a]\) must be separated by \(\Delta>0\) from the complementary eigenvalues of \(\widehat A\). Then, for operator or Frobenius norm,

\[
\|\sin\Theta(\widehat U,U)\|
\le \frac{\|EU\|}{\Delta}
\le \frac{\|E\|}{\Delta}.
\]

Using the enclosing interval matters; arbitrary pairwise eigenvalue distances are not a substitute for its separation requirement. [Davis–Kahan](https://doi.org/10.1137/0707001); [finite-dimensional statement and residual discussion](https://personal.lse.ac.uk/wangt60/publication/DKvariant.pdf)

Yu–Wang–Samworth Theorem 2 instead assumes only the reference gap

\[
\delta=\min\{\lambda_{a-1}-\lambda_a,\,
                 \lambda_b-\lambda_{b+1}\}>0,
\quad
\lambda_0=+\infty,\quad\lambda_{n+1}=-\infty.
\]

It gives

\[
\|\sin\Theta(\widehat U,U)\|_F
\le
\frac{2\min\{\sqrt q\,\|E\|_2,\|E\|_F\}}{\delta},
\]

and an orthogonal \(Q\) exists such that

\[
\|\widehat UQ-U\|_F
\le
\frac{2^{3/2}\min\{\sqrt q\,\|E\|_2,\|E\|_F\}}{\delta}.
\]

No small-\(E\) assumption is needed for these inequalities; large perturbations can make them uninformative. [Yu–Wang–Samworth, Theorem 2](https://personal.lse.ac.uk/wangt60/publication/DKvariant.pdf)

**Assessment of \(\varepsilon<\delta/4\).** For the leading \(r\)-dimensional subspace, define

\[
\delta=\lambda_r(A)-\lambda_{r+1}(A)>0,
\qquad \varepsilon=\|\widehat A-A\|_2.
\]

Weyl’s inequality gives

\[
\widehat\lambda_r-\widehat\lambda_{r+1}
\ge\delta-2\varepsilon.
\]

Consequently, \(\varepsilon<\delta/2\) already preserves a separated leading cluster. Combining mixed-gap separation with Davis–Kahan yields

\[
\|\widehat U\widehat U^\top-UU^\top\|_2
=\|\sin\Theta(\widehat U,U)\|_2
\le\frac{\varepsilon}{\delta-\varepsilon}.
\]

Thus **\(\varepsilon<\delta/4\) is valid but conservative**:

\[
\widehat\lambda_r-\widehat\lambda_{r+1}>\delta/2,
\qquad
\|\widehat P-P\|_2
\le \frac{4\varepsilon}{3\delta}<\frac13.
\]

Keep it if convenient; use \(\varepsilon<\delta/2\) if only cluster preservation and the displayed bound are needed. Neither threshold is required merely to invoke the Yu–Wang–Samworth inequality.

Three applicability details are essential:

- **Two learners:** if each is within \(\varepsilon\) of a common operator, their difference can be \(2\varepsilon\). Comparing each projector with the common reference gives
  \[
  \|\widehat P_1-\widehat P_2\|_2
  \le\frac{2\varepsilon}{\delta-\varepsilon}.
  \]
- **Degeneracy:** internal eigenvalue ties are allowed. Compare whole eigenspaces/projectors, allowing rotations within them. A cutoff through a tied eigenvalue has zero boundary gap and does not identify a unique subspace.
- **Correct operator:** define both error and gap in the same inner product and normalization. If the claim concerns \(h(A)\), use its relevant spectral separation; objective-dependent weights can reorder or collapse modes. General nonnormal transition matrices require different perturbation results, unless a valid self-adjoint/reversible transformation is supplied.

**Independent elementary propositions.**

1. **Spectral weighting preserves an eigenbasis, subject to ordering.**  
   If \(T=U\Lambda U^\top\) is symmetric and a learner’s kernel is *assumed or separately proved* to be \(K_j=h_j(T)\), then
   \[
   K_j=U\,\operatorname{diag}(h_j(\lambda_i))\,U^\top.
   \]
   This follows directly from spectral functional calculus. A shared leading subspace follows only when the selected indices remain the same under each \(h_j\); positive semidefiniteness additionally requires \(h_j(\lambda_i)\ge0\). This assumption cannot be inferred merely because an objective is contrastive.

   An entrywise transformation is different: for
   \[
   A=\begin{pmatrix}2&1\\1&1\end{pmatrix},
   \qquad
   \log_{\rm entrywise}A=
   \begin{pmatrix}\log2&0\\0&0\end{pmatrix},
   \]
   the matrices do not commute and do not share an eigenbasis. This independently rules out treating a generic entrywise log-occupancy ratio as a spectral logarithm.

2. **Bijective observations preserve operators; coarse observations generally compress them.**  
   Let \(Z\sim\mu\), \(X=h(Z)\), and \(J:L^2(\mu_X)\to L^2(\mu)\) be the pullback
   \(Jf=f\circ h\). Because \(\mu_X\) is the pushforward measure, \(J\) is an isometry. For a latent transition operator \(T\), the observed one-step operator is
   \[
   T_X=J^*TJ,
   \]
   by iterated conditional expectation. If \(h\) is bijective almost surely, \(J\) is unitary and this is unitary equivalence. Otherwise it is a compression: information outside the observable subspace can disappear. Spectral preservation then needs additional invariance assumptions. This argument does not establish bijective sensing in brains.

3. **Linear readout induces a PSD metric.**  
   For stimulus features \(Z\in\mathbb R^{n\times d}\) and readout \(B\),
   \[
   Y=ZB
   \quad\Longrightarrow\quad
   YY^\top=Z M Z^\top,\qquad M=BB^\top\succeq0.
   \]
   Conversely every PSD \(M\) has such a factorization. Therefore
   \[
   \|y_i-y_j\|^2=(z_i-z_j)^\top M(z_i-z_j).
   \]
   Diagonal \(M\) weights native axes; full \(M\) also mixes them. A fixed output dimension limits \(\operatorname{rank}(M)\). An unrestricted signed diagonal weighting need not be PSD.

   For ridge regression with unaveraged squared-error loss,
   \[
   \widehat Y=K(K+\alpha I)^{-1}Y,\qquad K=ZZ^\top,\quad\alpha>0.
   \]
   Feature rescaling changes \(K\) and generally changes this fit. In contrast, unregularized fitted values are the projection onto \(\operatorname{col}(Z)\), which an invertible feature transformation preserves.

4. **Measurement mixing can rotate, reorder or remove stimulus modes.**  
   Write latent features as \(Z=UD\). After linear measurement,
   \[
   YY^\top=U\,D BB^\top D\,U^\top.
   \]
   Off-diagonal entries of \(BB^\top\) mix latent modes; unequal diagonal entries reweight them; rank deficiency can eliminate them. Hence measured eigenvectors need not equal latent eigenvectors even with identical underlying representations.

   With \(Y^{(t)}=ZB+E^{(t)}\), independent zero-mean repeat noise gives
   \[
   \mathbb E[Y^{(1)}Y^{(2)\top}\mid Z]=ZBB^\top Z^\top.
   \]
   A same-repeat Gram includes noise covariance. A symmetrized cross-repeat estimate is unbiased under these assumptions but need not be PSD in a finite sample.

5. **Category pooling does not identify within-category structure.**  
   Let \(H\) be a nonempty-category indicator matrix and
   \[
   P=H(H^\top H)^{-1}H^\top.
   \]
   Then \(P=P^\top=P^2\), and
   \[
   Z=PZ+(I-P)Z,\qquad
   PKP=(PZ)(PZ)^\top,\quad K=ZZ^\top.
   \]
   Any residual change \(R\) satisfying \(PR=0\) leaves the pooled kernel unchanged. Such residuals can have arbitrarily different within-category geometry.

   Pooling **distances** requires a separate formula. For independent uniform draws from categories \(c,d\),
   \[
   \mathbb E\|Z_c-Z_d\|^2
   =\|\mu_c-\mu_d\|^2+
     \operatorname{tr}\Sigma_c+\operatorname{tr}\Sigma_d.
   \]
   Thus mean pairwise squared distance is generally not the squared distance between category means. Excluding identical within-category pairs changes the finite-sample factor.

6. **Latent geometry is not identifiable from an unrestricted readout alone.**  
   For every invertible \(R\),
   \[
   ZB=(ZR)(R^{-1}B).
   \]
   The observations are unchanged, whereas the latent Gram becomes
   \(ZRR^\top Z^\top\), generally different from \(ZZ^\top\). For example, \(Z=B=I_2\) and \(R=\operatorname{diag}(2,1/2)\) produce the same observed \(Y\) with different latent distances. Recovery of native geometry therefore requires restrictions on the representation/readout or additional observations.

7. **Eigenspace stability alone does not guarantee neighbour stability.**  
   Even an unchanged eigenbasis permits substantial distance changes through spectral reweighting. A sufficient direct condition is a distance margin: if
   \(\|\widehat K-K\|_2\le\eta\), then
   \[
   |\widehat d_{ij}^{\,2}-d_{ij}^{\,2}|
   =|(e_i-e_j)^\top(\widehat K-K)(e_i-e_j)|
   \le2\eta.
   \]
   Hence a squared-distance gap exceeding \(4\eta\) between an anchor’s \(k\)-th and \((k+1)\)-th neighbours preserves its kNN set. Without a margin or another explicit structural condition, Davis–Kahan does not by itself prove the proposed neighbourhood prediction.