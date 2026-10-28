# Hallucination source taxonomy (operational, heuristic)

The labels below describe *observable signal patterns*, not causes. `hallu.grounding.source.estimate_source`
returns `UNKNOWN` unless one of the rules fires; all thresholds are placeholders to be set on a validation split.

## Labels
| Label | Operational definition | Signal (heuristic) |
|---|---|---|
| Visual under-grounding | The hallucinated token was produced while the model attended little to image tokens. | R1: `image_attn_mass < low_image_attn` (0.10). |
| Language-prior influence | The token is about as likely without the image as with it, i.e. the text context alone explains it. | R2: `logprob(with image) - logprob(text only) < prior_gap` (1.0 nat). Needs a text-only forward pass. |
| Cross-modal conflict | The model attended to the image yet confidently asserted something the grounding check contradicts. | R3: grounding CONTRADICTED and `logprob > -0.5` and `image_attn_mass >= low_image_attn`. |
| Unknown | No rule fires or signals missing. | default |

Precedence: R3 > R2 > R1. Attention mass is a weak proxy for "use of the image"; a text-only logprob
gap conflates priors with prompt effects; COCO annotation contradiction is incomplete (annotation absence is not
object absence; only COCO classes absent from instances AND captions count). None of these are causal claims.

## Manual annotation subset
1. Sample ~200 hallucinated spans stratified by dataset and hallucination type (fixed seed), images available.
2. Two annotators independently label each span with one of the four labels, seeing image, prompt, response,
   the span, and (optionally) a text-only continuation; they do NOT see the heuristic output.
3. Guideline: choose *Cross-modal conflict* if a visible region plausibly misled the model (e.g. look-alike object);
   *Language-prior* if the span is a typical co-occurrence completion not supported by any region (e.g. "fork" next
   to a plate); *Visual under-grounding* if the span concerns a small/occluded/peripheral detail the model seemingly
   did not inspect; otherwise *Unknown*. When torn, pick *Unknown*.
4. Agreement: Cohen's kappa, kappa = (p_o - p_e) / (1 - p_e), with p_o observed agreement and
   p_e = sum_k p_A(k) p_B(k). Report kappa with a bootstrap 95% CI (1000 resamples over spans). Disagreements are
   adjudicated by discussion; adjudicated labels are used to evaluate the heuristic (per-class precision/recall).
5. Report kappa < 0.4 as "labels not reliable"; in that case no source-level claims are made.
