// Shared types live here so the four compressors never import each other.

export type ContentCategory = 'json' | 'log' | 'table' | 'text';

export interface CompressionResult {
  /** The text the model actually sees. */
  readonly text: string;
  /** e.g. `json:minify+sample`; goes into logs and metadata. */
  readonly strategy: string;
  /** Key facts of the run: strings truncated, repeats folded, and so on. */
  readonly notes: readonly string[];
}

/** Compressor signature; null means "not my kind of input". */
export type Compressor = (text: string, config: import('../config.ts').ResolvedConfig, estimator: import('../utils/tokenizer.ts').TokenEstimator) => CompressionResult | null;
