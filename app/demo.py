"""Local Gradio demo of the real pipeline (runs on this machine only; no share link).

LVLM generate -> token features -> detector -> calibrator -> selective correction.

    pip install -e .[model,demo]
    python app/demo.py --run-dir experiments/runs/main_mlp --calibrator temperature

If gradio, transformers, the model weights or the detector checkpoint are missing, the
demo shows a setup message. It never fabricates output. Risk values are MODEL ESTIMATES
from a learned detector, not ground truth.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SETUP = """### Setup required
{err}

1. `pip install -e .[model,demo]` (plus CUDA torch for GPU)
2. Train a detector: see `docs/reproduction.md` (needs features from the SAME LVLM)
3. `python app/demo.py --run-dir experiments/runs/<run> --calibrator temperature`
"""


class Pipeline:
    def __init__(self, a):
        from hallu.utils.config import load_config
        self.a = a
        self.cfg = load_config(a.model_config, a.overrides)
        self.lvlm = self.policy = None
        self.error = None
        try:
            self._load()
        except Exception as e:  # report, never fake
            self.error = f"{type(e).__name__}: {e}"

    def _load(self):
        from hallu.calibration.base import Calibrator
        from hallu.grounding.checker import GroundingChecker, VQABackend
        from hallu.mitigation.policies import SelectiveCorrection
        from hallu.mitigation.rewriter import Rewriter
        from hallu.models.lvlm import LVLM
        from hallu.utils.io import load_json
        from train_detector import load_detector
        run = Path(self.a.run_dir)
        for f in ("train_meta.json", "thresholds.json"):
            if not (run / f).exists():
                raise FileNotFoundError(f"detector checkpoint incomplete: {run / f} missing")
        meta = load_json(run / "train_meta.json")
        self.use = tuple(meta["features"])
        key = f"{self.a.calibrator}_seed{self.a.seed}"
        thr = load_json(run / "thresholds.json")[key]
        self.threshold = thr["f1_val" if self.a.threshold_type == "f1" else "risk_control_cal"]
        det = load_detector(run, meta["detector"], self.a.seed)
        cal = Calibrator.load(run / f"calibrator_{key}.pkl")
        lc = self.cfg["lvlm"]
        self.lvlm = LVLM(lc["model_id"], lc.get("device", "auto"), lc.get("dtype", "float16"),
                         lc.get("load_in_4bit", False), tuple(lc.get("hidden_layers", [-1])),
                         lc.get("output_attentions", False))
        self.policy = SelectiveCorrection(det, cal, self.threshold, GroundingChecker([VQABackend(self.lvlm)]),
                                          Rewriter(self.lvlm))
        self.config = {"model_id": lc["model_id"], "detector": meta["detector"], "features": list(self.use),
                       "calibrator": self.a.calibrator, "threshold": self.threshold,
                       "threshold_type": self.a.threshold_type, "seed": self.a.seed, "checker": "VQA (same LVLM)"}

    def run(self, image, prompt):
        if self.error:
            return SETUP.format(err=self.error), "", "", "", ""
        if image is None:
            return "Please upload an image.", "", "", "", ""
        from run_benchmarks import gen_features
        t0 = time.perf_counter()
        g = self.lvlm.generate(image, prompt or "Describe this image in detail.",
                               self.cfg["lvlm"].get("max_new_tokens", 128))
        t_gen = time.perf_counter() - t0
        X = gen_features(g, self.use)
        res = self.policy.apply(dict(response=g.text, prompt=prompt, image=image, offsets=g.offsets, X=X))
        probs = self.policy._probs(dict(X=X))
        spans = "".join(
            f'<span style="background:rgba(220,60,60,{min(float(p), 1):.2f})" title="risk {float(p):.2f}">'
            f"{html.escape(g.text[s:e])}</span>" + html.escape(g.text[e:(g.offsets[i + 1][0] if i + 1 < len(g.offsets) else e)])
            for i, ((s, e), p) in enumerate(zip(g.offsets, probs)))
        risk_html = ("<p><b>Token risk = model estimate</b> (calibrated detector output, not ground truth).</p>"
                     f"<div style='line-height:1.8'>{spans}</div>")
        verdicts = "\n".join(f"- `{d['text']}` risk={d['risk']:.2f} verdict={d['verdict']} action={d['action']}"
                             f" ({d['evidence']})" for d in res.decisions) or "No span above threshold."
        diff = "\n".join(f"- [{e.start}:{e.end}] `{e.old}` -> `{e.new}` ({e.reason})" for e in res.edits) or "No edits."
        lat = json.dumps({"generate_s": round(t_gen, 3), "correction_s": round(res.latency_s, 3),
                          **self.config}, indent=2)
        return f"**Original:** {g.text}\n\n**Corrected:** {res.corrected}", risk_html, verdicts, diff, lat


def main(argv=None):
    ap = argparse.ArgumentParser(description="Local demo")
    ap.add_argument("--model-config", default=str(ROOT / "configs" / "model.yaml"))
    ap.add_argument("--run-dir", default=str(ROOT / "experiments" / "runs" / "main_mlp"))
    ap.add_argument("--calibrator", default="temperature")
    ap.add_argument("--threshold-type", choices=["f1", "risk_control"], default="f1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    try:
        import gradio as gr
    except ImportError:
        sys.exit(SETUP.format(err="gradio is not installed (`pip install -e .[demo]`)."))
    pipe = Pipeline(a)
    with gr.Blocks(title="Selective hallucination correction (research demo)") as ui:
        gr.Markdown("## Risk-guided selective correction — research prototype\n"
                    "Risk scores are model estimates. Corrections may be wrong.")
        if pipe.error:
            gr.Markdown(SETUP.format(err=pipe.error))
        with gr.Row():
            img = gr.Image(type="pil", label="Image")
            prm = gr.Textbox(value="Describe this image in detail.", label="Prompt")
        btn = gr.Button("Run")
        o_text, o_risk = gr.Markdown(), gr.HTML()
        o_ver, o_diff = gr.Markdown(label="Verdicts"), gr.Markdown(label="Diff")
        o_cfg = gr.Code(label="Latency + config", language="json")
        btn.click(pipe.run, [img, prm], [o_text, o_risk, o_ver, o_diff, o_cfg])
    ui.launch(server_name="127.0.0.1", server_port=a.port, share=False)


if __name__ == "__main__":
    main()
