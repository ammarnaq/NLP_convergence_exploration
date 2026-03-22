# When Models Disagree, Language Reveals Itself

**Cross-method attribution convergence as evidence for latent pragmatic structure in natural language**



---

## The Core Insight

Machine learning interpretability has spent years asking **whether** attention explains model behaviour. This project asks a different question: **what do models agree on, and what does that agreement tell us about the structure of language itself?**

We train six models on SMS spam classification — from bag-of-words logistic regression to stacked BiLSTMs with attention — and extract token-level attributions from each. Using set-algebraic operations, we decompose their outputs into:

| Set | Definition | What it means |
|-----|-----------|---------------|
| **A ∩ R** | Tokens flagged by both attention *and* robust classical methods | **Corroborated signal** — multiple paradigms agree |
| **A \ R** | Tokens flagged by attention *only* | **Neural-exclusive signal** — what sequential context sees that statistics can't |
| **R \ A** | Tokens flagged by classical methods *only* | **Statistical-exclusive signal** — what frequency captures that attention ignores |

The finding: **these sets are not random.** They have consistent linguistic signatures that differ systematically between message classes.

---

## Why This Matters

### 1. Attribution disagreement is informative, not failure

The standard approach treats disagreement between attribution methods as noise — a reason to distrust one or both. We show that disagreement is **structured**: the tokens where methods diverge belong to identifiable linguistic categories, and those categories differ between ham and spam. Disagreement is signal.

### 2. Class-differential attribution structure

Ham messages yield smaller but more stable consensus sets than spam. The tokens that survive cross-method consensus in ham tend to be discourse markers and relational words ("hope", "ok", "sorry", "thanks"). In spam, consensus tokens are the expected lexical anchors ("free", "win", "claim"). This asymmetry — that different *types* of language produce different *patterns* of model agreement — has not been documented before.

### 3. Attention recovers pragmatically salient tokens invisible to classical methods

Attention-exclusive tokens (A \ R) on the ham side carry illocutionary weight: words that perform social actions (greetings, apologies, expressions of care) rather than transmit information. TF-IDF-based methods are structurally blind to these because their importance is contextual, not statistical. This means attention isn't hallucinating signal — it's detecting a **category of linguistic meaning** that classical attribution cannot access.

### 4. Small-scale convergence challenges scale-dependent explanations

Recent work on representational convergence — Huh et al.'s *Platonic Representation Hypothesis* (2024), Thasarathan et al.'s *Universal Sparse Autoencoders* (2025) — documents convergence across large models trained on massive data. Our pipeline produces analogous convergence phenomena with toy-scale models on 5,574 SMS messages. This suggests that cross-method convergence may be a property of **data structure and mathematical optimisation** rather than an emergent phenomenon of scale — a claim with implications for how we understand what models learn and why.

---

## Pipeline Architecture

```
SMS Dataset (5,574 messages)
│
├── TF-IDF + Logistic Regression ──► LR coefficients
├── TF-IDF + Random Forest ────────► Feature importances  
├── TF-IDF + Naive Bayes ──────────► Log-probability ratios
├── LSTM (mean-pool, no attention) ► Linear decomposition
├── BiLSTM + Attention ────────────► Attention weights
├── 2×BiLSTM + Attention ──────────► Attention weights
│
├── Set Algebra: A = ∩(attention models), R = consensus(classical, ≥2/4)
│   ├── A ∩ R  (corroborated)
│   ├── A \ R  (attention-exclusive)
│   └── R \ A  (classical-exclusive)
│
└── BART-MNLI Zero-Shot ───────────► Searle speech act taxonomy
    └── Per-token occlusion on pragmatic confidence
```

The ablation ladder is deliberately controlled: each model adds exactly one architectural variable (bag-of-words → sequence → bidirectionality → depth → attention), isolating what each capability contributes to attribution.

---

## Key Results

- **Ham/spam attribution asymmetry**: Ham consensus sets (|R|) are consistently non-empty; spam consensus sets are sparser. Models agree more on what makes a message *normal* than on what makes it *spam*.
- **POS-distribution divergence**: A ∩ R tokens in ham vs spam have statistically different grammatical compositions (permutation-tested), confirming that convergence tracks structurally different token types per class.
- **Attention-exclusive tokens carry pragmatic load**: A \ R ham tokens map to expressive and commissive speech acts under BART-MNLI classification — social language that performs actions rather than stating facts.
- **Discourse marker enrichment**: Ham A ∩ R tokens are enriched for discourse markers and affirmatives relative to a random permutation baseline.

---

## Implications

- * The models appeared to agree on tokens important to their task of spam classification using a quantifiable space of 'internal reasoning' on tokens that seemingly did not have lexical importance, suggesting instead that the models are relying on features that they were not trained to account for.
- * The 'space' of reasoning that even classical models are using to make a reasoning decision can be explained by natural language structure such as illocutionary intent.
- *  This implies that reasoning using pragmatic 'human' markers can be observable across different architectures of AI and is a property of the data itself.
- *  This project was heavily inspired by the Sufi concept of Nafs (نَفْس) which is an Arabic word occurring in the Quran literally meaning "self", and has been translated as "psyche", "ego" or "soul". The models in this pipeline were trained only to classify spam. What they learned to attend to reveals something about the structure of language that their training objective never asked for and that modern linguistic science may not be able to sufficiently capture A.I. behavior.
    
---

## Connections to Active Research

This work sits at the intersection of several live debates:

- **Attention as explanation** (Jain & Wallace 2019; Wiegreffe & Pinter 2019; Bastings & Filippova 2020) — we reframe from "is attention faithful?" to "what does attention uniquely capture?"
- **Platonic Representation Hypothesis** (Huh et al. 2024) — cross-method convergence at small scale as evidence for structure-driven rather than scale-driven representation alignment
- **Universal Sparse Autoencoders** (Thasarathan et al. 2025; Fel et al. 2025) — convergent feature extraction across architectures, which our set-algebra framework operationalises at the token level
- **Speech Act Theory** (Searle 1969) — illocutionary force as a linguistically principled characterisation of what neural attention detects beyond statistical salience

---

## Reproducibility

Built in a single Jupyter notebook. Python 3.12, PyTorch + TensorFlow/Keras, HuggingFace Transformers, NLTK, scikit-learn.

```bash
pip install torch tensorflow transformers nltk scikit-learn pandas matplotlib
```

The full pipeline — training, attribution extraction, set operations, statistical testing, and pragmatic analysis — runs end-to-end without external dependencies beyond the SMS Spam Collection dataset.

---

## Future Directions

- **Cross-method attribution disagreement as a subgroup signal for conformal prediction** — using the structure of A \ R vs A ∩ R to define subgroups for coverage auditing in uncertainty quantification. If models disagree about *which tokens matter* for certain inputs, those inputs may deserve different confidence guarantees.
- **Cross-lingual replication** — does the ham/spam attribution asymmetry hold in non-English SMS corpora? If discourse markers and relational tokens drive ham-side convergence universally, that's evidence for language-general pragmatic structure.
- **Scaling the framework** — applying the same set-algebra decomposition to transformer-scale models and more complex NLP tasks to test whether the convergence patterns observed here persist, strengthen, or dissolve.

---

## Citation

```
@misc{cross-method-attribution-convergence,
  title={Cross-Method Attribution Convergence Reveals Class-Differential 
         Pragmatic Structure in Text Classification},
  author={Ammar},
  year={2026},
  note={University of Dundee}
}
```

---

## Affiliation

Developed as part of MSc research at the **University of Dundee**, affiliated with the **Centre for Argument Technology (ARG-tech)**.

---

*Models don't just classify. They decompose language into what statistics can see and what only sequential context reveals. The boundary between those two is where pragmatics lives.*
