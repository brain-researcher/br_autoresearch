# EP23 co-occurrence operator: source feasibility audit (2026-10-13)

Produced by Codex `gpt-6-astra` (`max`, read-only, web search; no data) and reviewed by the orchestrator. The orchestrator verified the Visual Genome, LVIS, COCO and Open Images file sizes by download (see the experiment log). Coverage numbers in this audit are bounds or estimates; measured coverage is in `runs/operator_feasibility/`. "UNVERIFIED" marks unresolved items.

**Feasibility audit — 12 October 2026**

**Recommendation:** prioritize **full ADE20K annotations**, then **Visual Genome v1.4**, for visual-scene operators. Use **English Wikipedia** for a separate distributional text operator. All can be processed without downloading photographs.

This was a web-only audit. No shell commands, local-file reads, downloads, or benchmarks were performed. **UNVERIFIED** denotes information not established from the inspected sources. Published download sizes are distinguished from estimates.

Exact coverage of the 1,854 THINGS concepts—and your approximately 1,500-concept subset—is **UNVERIFIED for every external corpus**. Vocabulary size alone cannot establish coverage: synset identity, ambiguity, minimum frequency, and usable pair counts must also be checked.

| Rank | Recommended source | Reason | Main limitation |
|---|---|---|---|
| Visual 1 | Full ADE20K, 2021 index | Dense scene annotations; thousands of classes; supplied WordNet metadata; index already contains object-by-image counts | Access to the standalone index and its exact size need verification; rare THINGS concepts may remain unsupported |
| Visual 2 | Visual Genome v1.4 | Broad vocabulary, supplied synsets, very small annotation download | Dense but non-exhaustive; canonicalization and annotation errors |
| Text 1 | Pinned English Wikipedia dump | Broad vocabulary, public access, transparent text, manageable compressed size | Custom sense mapping; encyclopedic rather than visual co-occurrence |

These rankings are judgments from the evidence below, not measured coverage results. Visual Genome is the easiest immediate acquisition if ADE20K access delays the work.

**1. Visual Genome v1.2/v1.4**

**Access and files.** The [official project page](https://homes.cs.washington.edu/~ranjay/visualgenome/) and [official file index](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/) provide public downloads without registration. The posted license is CC BY 4.0.

| File | Published index size | Purpose |
|---|---:|---|
| `objects.json.zip` | 53M | v1.4 object annotations |
| `objects_v1_2.json.zip` | 70M | Earlier version, only if reproducing v1.2 |
| `object_synsets.json.zip` | 349K | Object-name canonicalization lookup |
| `object_synsets.json` | 1.2M | Uncompressed alternative |
| `object_alias.txt` | 59K | Synonym/alias groups |
| `image_data.json.zip` | 1.7M | Image metadata and cross-dataset identifiers |
| `synsets.json.zip` | 581K | Optional synset definitions/metadata |

These are rounded server listings; exact bytes and expanded JSON sizes are **UNVERIFIED**. The [v1.4 release note](https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/readme_v1_4.txt) explicitly identifies `objects.json.zip` as the cleaned object release and says image metadata and region descriptions remain unchanged from v1.2.

**Format and scale.** JSON groups objects by image; objects carry IDs, names, boxes, and synsets where supplied. Deduplicate mapped concepts within each image. The corpus has **108,077 images** and approximately **3.8 million object instances**. A full-corpus analysis reports **33,877 object labels**, which must not be interpreted as 33,877 distinct WordNet concepts. The exact distinct-synset vocabulary of the chosen dump is **UNVERIFIED**. [API schema](https://visualgenome.org/api/v0/api_readme), [author’s corpus analysis](https://cs.stanford.edu/~kenjihata/papers/Masters_Thesis.pdf).

**WordNet provenance and quality.** This is an official mapping produced by automatic canonicalization followed by human verification: noun-phrase extraction, WordNet’s most frequent matching sense, hand-written corrections for common mistakes, then crowd selection among candidate senses. The paper’s early audit reports imperfect object canonicalization; it is not a guarantee for every v1.4 annotation. [Visual Genome paper](https://arxiv.org/html/1602.07332).

**Exhaustiveness:** **No.** Region-driven annotation and verification establish observed positives, not verified absence of every unmentioned concept. No comprehensive negative-label mask repairs this. Broad THINGS coverage is plausible, but its extent is **UNVERIFIED**. COCO/Flickr identifiers support overlap checks; training-data independence cannot be presumed.

**2. LVIS v1**

**Access, terms, and files.** Use the [official dataset route](https://www.lvisdataset.org/dataset) and [official API repository](https://github.com/lvis-dataset/lvis-api). Annotation downloads are public; an author explicitly confirms **CC BY 4.0 for annotations**, with COCO’s image terms applying separately. [Author’s license clarification](https://groups.google.com/g/lvis-dataset/c/JeWfhD3vioM).

| File | Documented downloaded bytes |
|---|---:|
| [`lvis_v1_train.json.zip`](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip) | 350,264,821 |
| [`lvis_v1_val.json.zip`](https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip) | 64,026,968 |

These byte counts appear in a [TensorFlow download example](https://www.tensorflow.org/tfmodels/vision/instance_segmentation); current server-byte identity and expanded sizes are **UNVERIFIED**. No images are needed.

**Format and scale.** COCO-style JSON contains `images`, `annotations`, and `categories`; image records include federated-label fields. The usable public splits contain **100,170 training** and **19,809 validation images**, with **1,203 categories** overall. Categories supply an official WordNet synset, synonyms, and definition. The [category table](https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_categories.py) is also available separately. [Split documentation](https://www.tensorflow.org/datasets/catalog/lvis?authuser=86).

The images come from **COCO 2017**, but LVIS’s split membership is not simply COCO’s train/validation division.

**Exhaustiveness:** **Federated, not globally exhaustive.** `neg_category_ids` records verified absence; `not_exhaustive_category_ids` warns that annotation of a category’s instances is incomplete. Other unannotated categories are unknown. The correct pairwise treatment is specified below. [LVIS paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Gupta_LVIS_A_Dataset_for_Large_Vocabulary_Instance_Segmentation_CVPR_2019_paper.pdf).

**THINGS coverage:** with strict one-to-one synset matching, 1,203 categories impose a maximum of **64.9% of all THINGS concepts**, or approximately **80.2% of a 1,500-concept subset**, before actual intersection and frequency filtering. Actual coverage will be lower; expanding a broad category into multiple THINGS hyponyms would manufacture distinctions.

**3. ADE20K: full annotations versus SceneParsing-150**

**Use the full index, not the 150-class benchmark masks.** The [official ADE20K repository](https://github.com/CSAILVision/ADE20K) describes the 2021 full release as:

- **27,574 images:** 25,574 training and 2,000 held-out images.
- **3,688 object classes**, **707,868 object instances**, and **193,238 annotated parts**.
- **365 scene categories**.

The familiar SceneParsing benchmark instead uses **150 selected categories** and 20,210 training plus 2,000 validation images. It discards much of the vocabulary relevant here.

**Minimal acquisition.** The official documentation links [`index_ade20k.pkl`](https://ade20k.csail.mit.edu//toolkit/index_ade20k.pkl). Its [starter notebook](https://raw.githubusercontent.com/CSAILVision/ADE20K/main/notebooks/ade20k_starter.ipynb) documents:

- `objectPresence`: object-class-by-image instance counts.
- `objectIsPart`: corresponding part counts.
- Object names, filenames, folders, and scene labels.
- WordNet matching indicators, synset/hierarchy information, glosses, and synonyms.

Thus, the index can potentially supply the operator directly, without segmentation masks or photographs. Use binary presence; decide explicitly whether parts count as objects. Exact index bytes and successful current standalone delivery are **UNVERIFIED**. Full per-image JSON and segmentation/part PNGs are alternatives, but unnecessary if the index is complete.

**WordNet mapping:** officially supplied. The precise manual-versus-automatic construction and error rate of the **2021** crosswalk are **UNVERIFIED**; do not substitute claims about mappings from earlier releases.

**Access and license.** The [full-dataset request page](https://ade20k.csail.mit.edu/request_data/) requires registration/access handling. The posted [terms](https://ade20k.csail.mit.edu/terms/) distinguish restricted research/educational image use from **BSD-3-Clause annotation/software terms**. Whether the public index can currently be acquired independently of approval is **UNVERIFIED**.

**Exhaustiveness:** ADE20K aims at exhaustive annotation of visible scene objects and parts. This is stronger than VG’s region-derived inventory, but tiny, ambiguous, or unrecognized objects can still be missed. There is no per-class certified-negative audit. Broad THINGS coverage is plausible; exact coverage is **UNVERIFIED**.

**What Bonner & Epstein released.** Their 2021 study used an older ADE20K collection: **22,210 images**, 3,148 initial labels, and **2,959 cleaned categories**, with binary object presence and synonym/plural consolidation. They trained an eight-dimensional object2vec representation. The paper explicitly advertises **object2vec embeddings, stimuli, spatial-property ratings, preprocessed fMRI data, and analysis code** at [OSF](https://osf.io/ug5zd/). A separately released raw co-occurrence matrix, its filename, size, and license are **UNVERIFIED**; embeddings cannot reconstruct the original counts. [Bonner & Epstein, 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8253839/).

**4. Open Images V7**

**Access and required files.** The [official V7 download page](https://storage.googleapis.com/openimages/web/download_v7.html) provides these public, ungated annotation links:

- [`oidv7-train-annotations-human-imagelabels.csv`](https://storage.googleapis.com/openimages/v7/oidv7-train-annotations-human-imagelabels.csv)
- [`oidv7-val-annotations-human-imagelabels.csv`](https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv)
- [`oidv7-test-annotations-human-imagelabels.csv`](https://storage.googleapis.com/openimages/v7/oidv7-test-annotations-human-imagelabels.csv)
- [`oidv7-class-descriptions.csv`](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv)

Exact file sizes are **UNVERIFIED**. A few GB for human-label CSVs is a planning estimate, not a measured download size. The linked box-class hierarchy covers **600 boxable categories**, not the complete image-label vocabulary.

**Scale and format.** There are **9,178,275 images** overall. Human-verified image labels cover **7,337,077 training**, **41,620 validation**, and **125,436 test images**. The release reports approximately **61.4 million image-level labels** over **20,638 classes**. CSV columns include `ImageID`, `Source`, `LabelName`—a MID—and `Confidence`. [Official statistics](https://storage.googleapis.com/openimages/web/factsfigures_v7.html).

**Terms:** annotations are CC BY 4.0; image licensing is separate, generally listed as CC BY 2.0 with the project’s verification caveat.

**WordNet:** MIDs and English descriptions are supplied. A complete official MID-to-WordNet mapping is **UNVERIFIED**. Names alone are insufficient for sense matching; the vocabulary includes many entities outside THINGS.

**Exhaustiveness:** **No.** Human files contain verified positive and negative decisions, but unlisted labels remain unknown. Use `Confidence=1` and `Confidence=0` from the human files; exclude machine-generated label files. Human verification of machine-proposed candidates still creates selection bias. THINGS coverage could be broad, but mapped and adequately verified pair coverage is **UNVERIFIED**.

**5. COCO, COCO-Stuff, and COCO captions**

Use [COCO downloads](https://cocodataset.org/#download) and the [official COCO-Stuff repository](https://github.com/nightrome/cocostuff).

| Resource | Vocabulary and scale | Annotation-only files |
|---|---|---|
| COCO instances | **80 observed thing categories**; 118,287 train and 5,000 validation images in 2017 | `annotations_trainval2017.zip`, published as **241 MB**; use `instances_train2017.json` and `instances_val2017.json` |
| COCO-Stuff | **80 things + 91 stuff = 171 observed categories** | `stuff_trainval2017.zip`, **543 MB**; combine with COCO instance JSONs |
| COCO captions | Approximately five captions per image; approximately 616,000 train/validation captions; open vocabulary | `captions_train2017.json` and `captions_val2017.json`, inside the same 241 MB archive |

The Stuff label numbering includes unused thing IDs; **182 IDs including void is not 182 observed semantic classes**. Optional `stuffthingmaps_trainval2017.zip` masks are listed as 659 MB and are unnecessary for annotation-derived presence. Individual expanded JSON sizes are **UNVERIFIED**.

Annotations are CC BY 4.0, publicly downloadable without registration; image rights remain separate. JSON supplies category IDs or caption strings, not an official comprehensive WordNet crosswalk.

**Exhaustiveness:** instance/stuff annotation aims at coverage within its restricted taxonomy, subject to crowds, void regions, and missed instances. Captions are not exhaustive object inventories.

**Suitability:** too coarse as the principal THINGS operator. They are useful controls and implementation checks. COCO captions offer broader lexical coverage, but exact THINGS coverage is **UNVERIFIED**.

**6. Caption corpora**

All corpora below have open textual vocabularies rather than fixed THINGS-compatible concept inventories. None supplies a verified, complete caption-token-to-THINGS synset mapping. Captions describe selected content; omission does not verify visual absence.

| Source | Text-only acquisition, scale, and size | License/access and suitability |
|---|---|---|
| **Conceptual Captions / CC3M** | Training and validation caption–URL TSVs; **3,318,333 train**, **15,840 validation** pairs. Historical filenames are `Train_GCC-training.tsv` and `Validation_GCC-1.1.0-Validation.tsv`; current official manifest and exact sizes **UNVERIFIED**. Budget approximately **0.6 GB**, an estimate. | Public [Google download route](https://ai.google.com/research/ConceptualCaptions/download). The [official license](https://github.com/google-research-datasets/conceptual-captions/blob/master/LICENSE) is custom permissive language, not a CC license. Human web alt text was automatically filtered and transformed, including conceptual abstraction. |
| **CC12M** | `cc12m.tsv`, **approximately 12 million** URL–caption pairs; official repository advertises **2.5 GB**. URL precedes caption, unlike common CC3M exports. | Public [official repository](https://github.com/google-research-datasets/conceptual-12m), with a [custom permissive license](https://github.com/google-research-datasets/conceptual-12m/blob/main/LICENSE). Larger and less aggressively cleaned than CC3M. Do not acquire optional image hashes or pixels for this task. |
| **Localized Narratives** | Use the **“Textual captions only”** split downloads, not full traces/audio. **849,000 images** across Open Images, COCO, Flickr30k, and ADE20K; some images have multiple narratives. JSONL identifies dataset, image, annotator, and caption. Exact export basenames and sizes **UNVERIFIED**; likely hundreds of MB rather than tens of GB. | Public, **CC BY 4.0**. Crucially, final captions are **manually transcribed**; separately supplied automatic transcripts are for alignment. Richer scene descriptions make this a strong small caption-based alternative. [Official project/download page](https://google.github.io/localized-narratives/) |
| **RedCaps v1.0** | `redcaps_v1.0_annotations.zip`; JSON files organized by subreddit/year contain captions, original captions, URLs, and post metadata. An author-maintained release records **12,011,121 examples** and **1,061,908,181 downloaded bytes**. | Public [official downloader repository](https://github.com/redcaps-dataset/redcaps-downloader); dataset metadata lists **CC BY 4.0**. Original Reddit/image rights remain separate. Captions are human post titles, with strong subreddit and selection biases. [Author’s release metadata](https://huggingface.co/datasets/kdexd/red_caps/commit/7c89d87bdd296d555460b16ad781e93462b0ebe3) |

CC3M reports a token vocabulary under its preprocessing, but that is not a concept vocabulary. Exact lexical vocabulary sizes for these releases under the proposed tokenizer, and their THINGS coverage, are **UNVERIFIED**. Concept matching should preserve multiword names and reject ambiguous matches rather than automatically choosing WordNet’s first sense.

**LAION after the December 2023 takedown.** LAION took its datasets offline on **19 December 2023** following the identification of links to illegal material. On **30 August 2024**, it announced **Re-LAION-5B**, removing known offending links using updated lists and providing research/research-safe variants. Filtering is not certification that every remaining URL is safe. Metadata is released under **Apache 2.0**, while underlying content rights remain separate; the announcement recommends research use and discourages industrial/end-product deployment. [Official re-release announcement](https://laion.ai/blog/relaion-5b/).

The current [Re-LAION-2B English research-safe page](https://huggingface.co/datasets/laion/relaion2B-en-research-safe) exists, is gated by login/contact/terms acceptance, and lists **467 GB** of Parquet data. This is metadata, not a requirement to download image pixels, but it still exceeds the proposed budget. Exact accessible shard names, present record count, and manual-approval requirements are **UNVERIFIED**.

Current direct availability of the original **LAION-400M** and original **LAION-2B** exports is **UNVERIFIED**; historical download instructions do not establish current access. The original LAION-5B release contained approximately 5.85 billion pairs, including 2.32 billion English pairs. [Original release description](https://laion.ai/blog/laion-5b/).

**Recommendation:** do not use LAION for the first operator. Its size, access conditions, and CLIP-based selection complicate this audit. Human alt text selected by CLIP is also less independent of CLIP than independently collected human annotations. Generated recaptioning releases are a separate provenance category.

**7. Distributional text and behavioral association resources**

| Resource | Acquisition and size | Scale, format, mapping, and limitations |
|---|---|---|
| **English Wikipedia** | The inspected [official dump index](https://dumps.wikimedia.org/enwiki/latest/) lists `enwiki-latest-pages-articles-multistream.xml.bz2` at **26,917,225,724 bytes**. Public, no registration. Stream decompression; expanded text/XML can exceed 50 GB. | XML/wikitext covering millions of articles; exact article and vocabulary counts for that snapshot **UNVERIFIED**. Keep main-namespace non-redirect articles; define sentence, paragraph, or fixed-window units. No supplied WordNet mapping. Text is principally CC BY-SA 4.0, with applicable legacy provisions. [Copyright terms](https://en.wikipedia.org/wiki/Wikipedia:Copyrights) |
| **Google Books Ngram** | Public [official dataset exports](https://storage.googleapis.com/books/ngrams/books/datasetsv3.html), including the 2020 release; gzip TSV shards, CC BY 3.0. Exact desired shard sizes and total acquisition size **UNVERIFIED**. | Aggregated 1–5-grams with year, occurrence count, and volume count; dependency exports also exist. No document identities or arbitrary sentence contexts. **Unigrams cannot produce pair co-occurrence**, and n-grams cannot reconstruct general book-level co-occurrence. Vocabulary/THINGS intersection **UNVERIFIED**. |
| **COCA** | [Official download products](https://www.english-corpora.org/coca/help/download.asp) include full text and collocate tables, under paid/custom licensing. Exact files, sizes, and current price **UNVERIFIED**. | Approximately **1 billion words**, about **485,000 texts**, 1990–2019. Published collocates cover roughly **13.5 million pairs**, taking top collocates for approximately **60,000 lemmas**. These are truncated tables: an absent pair need not have zero count. No official THINGS synset mapping. [Corpus composition](https://www.english-corpora.org/coca/help/texts.asp) |
| **GloVe resources** | The [official project](https://nlp.stanford.edu/projects/glove/) distributes vectors, including the 2014 Wikipedia/Gigaword package: **822 MB**, 400,000-word vocabulary. | These are **embeddings, not a raw co-occurrence matrix**. Training code includes a co-occurrence builder. A ready-made raw matrix matching THINGS is **UNVERIFIED**. Do not substitute vector similarity for counts without changing the operator’s definition. |
| **SWOW-English** | [Official research downloads](https://smallworldofwords.org/en/project/research); exact current filenames and bytes **UNVERIFIED**. Posted terms are **CC BY-NC-ND 3.0**; derived-data redistribution needs attention to those terms. | **Behavioral association**, not distributional co-occurrence. The published cleaned dataset has **12,217 cues**, **83,863 participants**, and **3,665,100 responses**; 12,292 is an earlier/raw cue count. No supplied THINGS WordNet mapping. [Authors’ paper](https://simondedeyne.me/articles/2019.DeDeyne.Small%20World%20of%20Words%20Engish%20association%20norms.BehaviorResearchMethods.pdf) |

Wikipedia offers the strongest general lexical coverage expectation, but exact sense-resolved coverage remains **UNVERIFIED**. Pin a dated snapshot and checksum. If pre-generative-text provenance is essential, prefer a pre-2023 snapshot; its currently available file and size require separate verification.

SWOW naturally yields a **directed cue-to-response matrix**. Symmetrizing counts or response probabilities produces different behavioral operators; specify which one is intended. An unelicited association is not a verified zero association.

**8. Egocentric and child-headcam sources**

**Ego4D.** The [official access instructions](https://ego4d-data.org/docs/start-here/) require a license application and approved credentials; they advertise narration-only acquisition at approximately **350 MB**. That is a documented planning size, not a verified size for the latest complete narration release. The original corpus is approximately **3,600 hours from 926 participants**; the exact narration/document count for the intended version is **UNVERIFIED**.

Narration JSON groups timestamped text and summaries by video and annotation pass. An [official maintainer discussion](https://discuss.ego4d-data.org/t/download-of-only-narration-json-from-the-annotations-folder/543) identifies v2 `narration.json` and recommends the more comprehensive `all_narrations_redacted.json`; the latter’s exact size is **UNVERIFIED**. [Schema documentation](https://ego4d-data.org/docs/data/annotations-schemas/).

Narrations are action-focused and non-exhaustive. No fixed noun vocabulary or official complete WordNet crosswalk was verified. Define temporal windows and deduplicate annotation passes. The [consortium license document](https://ego4d-data.org/pdfs/Ego4D-Licenses-Draft.pdf) contains custom, purpose-limited permissions and redistribution restrictions; it should not be summarized as a generic CC or blanket noncommercial license.

**EgoObjects.** The [official repository](https://github.com/facebookresearch/EgoObjects) reports approximately **114,000 frames**, **9,000 videos**, **250 participants**, **14,400 object instances**, and **368 categories**. Needed annotations are `EgoObjectsV1_unified_train.json`, `EgoObjectsV1_unified_eval.json`, and `EgoObjectsV1_metadata.json`; exact byte sizes are **UNVERIFIED**. The approximately 40 GB image acquisition is unnecessary.

Annotations are LVIS-like instance/box JSON. Annotators seek exhaustive boxes for the discovered category inventory, but this is not an unrestricted inventory of all possible objects. Frames from the same object/video are strongly dependent. Official WordNet mapping is **UNVERIFIED**. [Dataset paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_EgoObjects_A_Large-Scale_Egocentric_Dataset_for_Fine-Grained_Object_Understanding_ICCV_2023_paper.pdf).

The [Meta dataset agreement](https://ai.meta.com/datasets/egoobjects-downloads/) governs the data; repository code licensing does not replace it. The agreement lists specific research purposes. Applicability to this co-occurrence analysis and current approval mechanics are **UNVERIFIED**. The 368-category ceiling makes this a specialized corroboration source.

**EPIC-KITCHENS-100.** The [official annotation repository](https://github.com/epic-kitchens/epic-kitchens-100-annotations) supplies `EPIC_100_train.csv`, `EPIC_100_validation.csv`, and noun-class tables. Exact file sizes are **UNVERIFIED**, but these are small tabular annotations, not video downloads. Public annotations use **CC BY-NC 4.0**.

The original benchmark covers **100 hours**, approximately **90,000 action segments**, **97 verbs**, and **300 noun classes**; later corrected class files must be version-pinned. CSV rows contain narration text, timestamps, and action labels. No official complete WordNet mapping was verified.

Crucially, an action’s noun label is **not an exhaustive frame inventory**. One noun per action cannot directly produce within-frame object pairs: temporal-window or narration-text co-occurrence would define a different operator. Automatically generated boxes/masks need separate model-provenance scrutiny.

**SAYCam.** Access is through [Databrary volume 564](https://nyu.databrary.org/volume/564), requiring authorized researcher/institutional access. The corpus follows **three children**, approximately 6–32 months old, over roughly **500 hours**. The original publication reports over **200,000 transcribed words**, with transcription continuing. Video summaries also describe salient objects, locations, and activities. [SAYCam paper](https://direct.mit.edu/opmi/article/doi/10.1162/opmi_a_00039/97495/SAYCam-A-Large-Longitudinal-Audiovisual-Dataset).

Exact transcript/summary filenames, text-only sizes, currently transcribed coverage, vocabulary, and WordNet mapping are **UNVERIFIED**. Labels and utterances are not exhaustive visual inventories. Databrary access and sharing terms apply; open-access publication does not make the underlying child recordings unrestricted.

**BabyView.** The [official dataset page](https://babyview-project.github.io/dataset/) describes institutional/Databrary access and restrictions on identification and redistribution. Releases have expanded: the original paper described approximately **493 hours**, while a [2025 report](https://openreview.net/pdf?id=bbv4CA2noZ) describes **868 home-recording hours from 31 children**. Do not mix release totals.

Exact downloadable text-only files, bytes, current transcript counts, object vocabulary, and WordNet crosswalk are **UNVERIFIED**. The project uses automatic speech processing, with a much smaller human-checked subset. Consequently, transcript-derived counts require explicit ASR provenance and error assessment; they are not an exhaustive human-labeled object inventory. [BabyView methods](https://arxiv.org/html/2406.10447v2).

For all five sources, exact THINGS coverage is **UNVERIFIED**. Less curated recording does not itself establish independence from models trained on egocentric datasets.

**9. THINGS-specific resources**

**THINGSplus contains potentially useful context labels.** Beyond property norms and **53 higher-level categories**, the authors asked participants to name the prominent object and **other objects in all 26,107 THINGS images**. These human labels could support an image-context operator. They are not merely main-object labels. [THINGSplus paper](https://link.springer.com/article/10.3758/s13428-023-02110-8).

The official release is [OSF `jum2f`](https://osf.io/jum2f/). Exact secondary-object label filenames, bytes, data license, and a synset-normalized secondary vocabulary are **UNVERIFIED**. Instructions to list all objects do not establish certified negative labels.

All 1,854 main concepts are represented, but **usable secondary-object coverage and pair support are UNVERIFIED**. Moreover, these are object-selected THINGS photographs, potentially overlapping the evaluated stimulus pool. They estimate context in that stimulus collection, not an independent sample of natural scene frequencies.

**No verified published 1,854 × 1,854 natural-scene count matrix was found.** This is a bounded literature-search result, not proof of nonexistence. In particular:

- Bonner & Epstein’s released object2vec embeddings are not established here as a released raw count matrix.
- [Magri et al., 2023](https://www.bonnerlab.org/papers/scene_context.pdf) analyzes selected sets of **81 ADE20K objects** and **77 COCO objects**, with further subsets for some analyses—not all THINGS concepts. Its model-derived contextual prototypes must also be distinguished from annotation counts.

**SPoSE/THINGS behavioral embeddings are available.** The [THINGS initiative](https://things-initiative.org/) links the [behavioral release](https://osf.io/f5rn6/), including the **66-dimensional representation of 1,854 concepts** derived from approximately **4.7 million odd-one-out judgments**. This differs from the earlier 49-dimensional release.

Exact archive filename, byte size, and data-license details are **UNVERIFIED**. The numeric 1,854 × 66 float64 array alone would occupy approximately **0.98 MB**, calculated rather than downloaded. This is **behavioral similarity**, not scene co-occurrence or free word association. GPT-generated THINGS semantic features are another, model-generated resource and do not have the same provenance.

**10. Verified-label counting and training overlap**

For LVIS, define for image \(i\):

- \(P_i\): categories with an annotated positive instance.
- \(N_i\): `neg_category_ids`.
- \(Q_i\): `not_exhaustive_category_ids`.

For **binary presence**, an annotated instance proves presence even when its category is in \(Q_i\). A suitable known-status set is therefore \(K_i=P_i\cup N_i\). For an additionally conservative analysis restricted to exhaustively annotated positives, use \(K_i=(P_i\setminus Q_i)\cup N_i\). Do not infer presence solely from a warning flag.

For each pair, retain its joint verification denominator:

\[
E_{cd}=\sum_i \mathbf1[c,d\in K_i],\qquad
A^{\mathrm{verified}}_{cd}
=\sum_{i:c,d\in K_i}x_{ic}x_{id}.
\]

Compute marginals on the **same jointly verified images**. For Open Images, construct \(K_i\) from explicit human positive/negative decisions.

**Neither procedure recovers unbiased population co-occurrence automatically.** Categories are not selected for verification randomly. Pair-specific \(A_{cd}/E_{cd}\) values also need not form one coherent global joint distribution or a positive-semidefinite matrix. Where \(E_{cd}=0\), the pair is unknown.

Three independence questions should remain separate:

| Question | Audit conclusion |
|---|---|
| Did an evaluated representation generate the annotation? | Human VG/LVIS/ADE labels and human captions avoid direct generation by that representation. Automatic labels, generated captions, and some ASR do not establish this. |
| Did the evaluated model see the same images/text during training? | Requires the actual checkpoint and training manifest. Human annotation provenance alone cannot answer it. |
| Are purportedly separate source samples independent? | COCO, LVIS, COCO-Stuff, parts of VG, and Localized Narratives share images. They must not be treated as independent replications. |

A concrete DINOv2 qualification: its paper lists **ADE20K as a retrieval source when assembling LVD-142M**, rather than simply including ADE20K “as is.” I did not find support in its dataset table for claiming that COCO was explicitly included. Exact incidental COCO/LVD-142M overlap is **UNVERIFIED**. [DINOv2, Tables 15 and 18](https://arxiv.org/html/2304.07193v2).

LAION exposure is documented for particular [OpenCLIP checkpoints](https://github.com/mlfoundations/open_clip). This does not establish that OpenAI’s original CLIP used LAION, nor quantify COCO’s overlap with either training corpus. Exact overlap for VG, Open Images, caption corpora, Wikipedia, egocentric datasets, and the unspecified evaluated checkpoints remains **UNVERIFIED**.

**11. Normalization and eigenvectors**

Start with binary image/document presence \(B_{ic}\), deduplicated after synonym mapping:

\[
A=B^\top B.
\]

Keeping its diagonal gives \(A_{cc}=f_c\), the concept frequency, and preserves the Gram-matrix interpretation. Zeroing the diagonal creates a different, potentially indefinite adjacency operator.

| Transformation | Preserves eigenvectors of the original symmetric \(A\)? | Explanation |
|---|---|---|
| Genuine matrix function \(f(A)=Uf(\Lambda)U^\top\) | **Yes**, a common eigenbasis | Includes matrix powers, matrix exponential, and admissible spectral shrinkage. Eigenvalue ordering can change, so the *leading* selected subspace need not remain the same. |
| Entrywise `log1p(A)` | **No, generally** | Applies a scalar transform to entries, not eigenvalues. It is not the matrix logarithm. |
| PMI / PPMI | **No, generally** | Marginal corrections, entrywise logarithms, and entrywise clipping alter eigenvectors. Nonnegative entries do not imply positive semidefiniteness. |
| \(S=D^{-1/2}AD^{-1/2}\) | **No, generally** | A symmetric congruence transformation, not ordinarily a spectral function of \(A\). |
| \(P=D^{-1}A\) | **No, generally** | Usually nonsymmetric; related to \(S\) by similarity, not identical eigenvectors. |
| \(I-S\) | **Yes relative to \(S\)** | Its eigenvalues are \(1-\lambda(S)\), reversing the relevant ordering. |

Here \(D=\operatorname{diag}(A\mathbf1)\), unless a different degree definition is explicitly intended. For positive degrees,

\[
P=D^{-1/2}SD^{1/2}.
\]

Thus \(P\) and \(S\) share eigenvalues; if \(Su=\lambda u\), then \(D^{-1/2}u\) is a right eigenvector of \(P\). Use \(S\) when ordinary Euclidean-orthonormal eigenvectors are required. [Chung, *Spectral Graph Theory*](https://mathweb.ucsd.edu/~fan/research/cb/ch1.pdf), [von Luxburg, 2007](https://arxiv.org/abs/0711.0189).

**Specify PMI’s sampling space.** For complete binary presence over \(N\) images,

\[
\operatorname{PMI}_{cd}
=\log\frac{N A_{cd}}{f_cf_d}.
\]

For a matrix of sampled token/context pairs, the conventional formula instead uses total pair mass and row/column sums:

\[
\operatorname{PMI}_{cd}
=\log\frac{A_{cd}\sum_{ab}A_{ab}}
 {(\sum_bA_{cb})(\sum_aA_{ad})}.
\]

These are different estimands. PPMI is \(\max(0,\operatorname{PMI})\); sparse counts require a declared zero/smoothing policy. [Church & Hanks, 1990](https://aclanthology.org/J90-1003/), [Levy & Goldberg, 2014](https://proceedings.neurips.cc/paper_files/paper/2014/file/b78666971ceae55a8e87efb7cbfd9ad4-Paper.pdf).

Exclude or separately record zero-degree concepts. Specify largest-algebraic versus largest-magnitude eigenvalues. Also distinguish direct co-occurrence \(A\) from shared-context similarity such as \(AA^\top\).

**12. CPU feasibility and practical pitfalls**

The estimates below are **UNVERIFIED engineering estimates**, excluding downloads, manual mapping, and neural disambiguation. They assume streaming parsers, deterministic phrase matching, local storage, and counting only mapped THINGS concepts.

| Workload | Approximate counting cost |
|---|---:|
| ADE index, VG, LVIS, COCO/Stuff, EgoObjects | Minutes to about 1 CPU-hour each |
| Open Images human-label CSVs | About 1–4 CPU-hours |
| CC3M / CC12M | About 0.5–4 CPU-hours |
| Localized Narratives | Minutes to about 1 CPU-hour |
| RedCaps | About 1–6 CPU-hours |
| Full English Wikipedia | About 10–100 CPU-hours, strongly parser-dependent |
| Ego4D narrations / EPIC / available child transcripts | Minutes to a few CPU-hours |
| Published collocate/association tables | Usually minutes after acquisition |
| Full Re-LAION English metadata | Not recommended within the acquisition budget |

A dense **1,854 × 1,854 float64 matrix is only 27.5 MB**; a 1,500-concept matrix is 18 MB. Counting and parsing dominate memory if implemented carelessly. Leading eigenspaces should be inexpensive relative to corpus processing; seconds to minutes is a reasonable expectation, not a benchmark. An initial Slurm plan of 4–8 CPUs and 16–32 GB RAM is ample for most annotation sources with streaming.

| Pitfall | Mitigation |
|---|---|
| Frequency dominates eigenvectors | Report concept frequencies/degrees; compare raw, symmetric-normalized, and PPMI operators; use frequency-preserving nulls |
| Unknown annotations treated as absence | Preserve positive/negative/unknown status and pairwise verification denominators |
| Synonym splitting or polysemy | Pin WordNet version; audit aliases and multiword matches; keep ambiguous cases unresolved |
| Broad labels expanded into specific THINGS concepts | Do not copy a parent-category occurrence into every child concept |
| Repeated instances inflate counts | Use binary presence for the proposed image-level operator; analyze instance counts separately |
| Scene, photographer, subreddit, household, or video imbalance | Stratify or reweight where justified; bootstrap at the independent image/source/video/person unit |
| Rare pairs receive extreme PPMI | Prespecify minimum support and smoothing; report sensitivity and unsupported pairs |
| Duplicate images or overlapping corpora | Deduplicate IDs/URLs where possible; treat shared-image sources as alternate annotations |
| Captions, documents, and temporal windows imply different relations | Prespecify the context unit; do not call all resulting matrices visual co-occurrence |
| Mapping or normalization tuned against fMRI results | Freeze construction using external data before evaluating representations |
| Subsetting changes normalization and eigenspaces | Prespecify whether normalization occurs on all 1,854 concepts, the approximately 1,500-concept union, or each participant’s subset; these operations do not commute |

The first reviewable artifact should be a **synset crosswalk and coverage table** for ADE20K and VG: mapped concepts, unresolved senses, frequencies, supported pairs, and coverage for each participant. That establishes whether either source supports the intended analysis before committing to larger corpora.