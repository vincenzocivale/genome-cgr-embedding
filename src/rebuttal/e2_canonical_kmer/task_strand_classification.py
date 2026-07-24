"""
Rebuttal E2 — per-task strand-invariance classification.

Reviewer HnDT asks whether the k-mer baseline's strand-specific (forward-strand
only) encoding is biologically appropriate, and whether results are robust to
canonical (reverse-complement-aware) k-mers. Canonicalization should only be
applied where task labels do not depend on strand orientation; forcing it on
strand-specific tasks would discard real biological signal.

Classification below is per dataset-category (companion to `infer_biocat` in
src/analysis/paper_utils.py), assigned by inspecting each benchmark's biology, not
by any statistical criterion:

  strand_specific — orientation carries the label's meaning; do NOT canonicalize.
    * splice/*                                  GT-AG donor/acceptor sites are
                                                  defined on the coding strand.
    * prom/*, iPro-WAEL/Promoter_*,
      genomic_benchmark/promoter_notata_251bps   promoters drive transcription in
                                                  one direction (TSS is directional).
    * genomic_benchmark/coding                   coding-region detection depends on
                                                  reading frame / strand.
    * iDNA_ABF/{5mC,6mA}, deep4mc/*_4mC          the modified base sits on a
                                                  specific strand.
    * virus/covid_variants                       SARS-CoV-2 is a defined-polarity
                                                  positive-sense ssRNA virus.

  strand_invariant — orientation is not part of the label; canonicalization is
  biologically defensible.
    * EMP/*                                      histone marks reflect chromatin
                                                  state, not strand.
    * enhancers/*,
      genomic_benchmark/{enhancer_cohn,enhancer_ensembl}
                                                  enhancers are classically
                                                  orientation-independent.
    * iDHS-EL/DNase_I,
      genomic_benchmark/open_chromatin_region     chromatin accessibility does not
                                                  depend on strand.
    * tf/Human_TFBS_*, mouse/mouse_TFBS_*         double-stranded DNA-binding
                                                  proteins recognize the motif on
                                                  either strand.

  ambiguous — default to NOT canonicalizing (conservative), reported separately.
    * genomic_benchmark/human_vs_worm             species-origin classification;
                                                  no clear directional biology.
    * genomic_benchmark/regulatory_region_type     mixes promoter-like and
                                                  enhancer-like elements; strand
                                                  relevance depends on the specific
                                                  element type within the task.
"""

STRAND_SPECIFIC_PREFIXES = (
    "splice/",
    "prom/",
    "iPro-WAEL/Promoter_",
    "genomic_benchmark/promoter_notata_251bps",
    "genomic_benchmark/coding",
    "iDNA_ABF/5mC",
    "iDNA_ABF/6mA",
    "deep4mc/",
    "virus/",
)

STRAND_INVARIANT_PREFIXES = (
    "EMP/",
    "enhancers/",
    "genomic_benchmark/enhancer_cohn",
    "genomic_benchmark/enhancer_ensembl",
    "iDHS-EL/",
    "genomic_benchmark/open_chromatin_region",
    "tf/Human_TFBS_",
    "mouse/mouse_TFBS_",
)

AMBIGUOUS_PREFIXES = (
    "genomic_benchmark/human_vs_worm",
    "genomic_benchmark/regulatory_region_type",
)


def infer_strand_invariance(dataset: str) -> str:
    """Return one of 'strand_specific', 'strand_invariant', 'ambiguous' for a dataset name."""
    if dataset.startswith(AMBIGUOUS_PREFIXES):
        return "ambiguous"
    if dataset.startswith(STRAND_SPECIFIC_PREFIXES):
        return "strand_specific"
    if dataset.startswith(STRAND_INVARIANT_PREFIXES):
        return "strand_invariant"
    raise ValueError(f"Unclassified dataset for strand invariance: {dataset!r}")


CANONICALIZATION_ELIGIBLE = {"strand_invariant"}


def canonicalization_applies(dataset: str) -> bool:
    """Whether canonical (RC-aware) k-mers are biologically appropriate for this task."""
    return infer_strand_invariance(dataset) in CANONICALIZATION_ELIGIBLE
