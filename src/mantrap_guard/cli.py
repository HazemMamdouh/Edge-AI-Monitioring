"""`mantrap` command line interface.

    mantrap annotate   draw the mantrap polygon, entry line and exit line on a video frame
    mantrap run        run the pipeline on a video, write the visualisation + metrics
    mantrap export     convert / quantise the models: ONNX, TensorRT FP16, TensorRT INT8
    mantrap benchmark  speed + agreement table for every model variant
    mantrap compare    merge benchmarks from several devices into one table
    mantrap render     redraw a run's video from detections.jsonl, optionally with new rules
    mantrap showcase   compose input | output | system stats | terminal into one video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import __version__
from .config import load_config

console = Console()


def _annotate(a):
    from .annotate import annotate_headless, annotate_interactive

    if a.mantrap or a.entry_line or a.exit_line:
        if not (a.mantrap and a.entry_line and a.exit_line):
            sys.exit("headless mode needs --mantrap, --entry-line and --exit-line together")
        annotate_headless(a.video, a.out, a.mantrap, a.entry_line, a.exit_line, a.frame, a.preview)
    else:
        z = annotate_interactive(a.video, a.out, a.frame, a.preview)
        if z is None:
            sys.exit("nothing saved (press s to save before quitting)")
    console.print(f"[green]zones saved[/] {a.out}" + (f", preview {a.preview}" if a.preview else ""))


def _run(a):
    from .pipeline import run_pipeline

    over = {"model": {}}
    for key, val in (("pose", a.model), ("objects", a.objects_model), ("imgsz", a.imgsz),
                     ("device", a.device)):
        if val is not None:
            over["model"][key] = val
    if a.half:
        over["model"]["half"] = True
    if a.flow:
        over["flow"] = {"method": a.flow}
    cfg = load_config(a.config, over)
    run_pipeline(a.video, a.zones, a.out, cfg, max_frames=a.max_frames, start=a.start,
                 show=a.show, dashboard=a.dashboard)
    if a.showcase:
        from .showcase import build_showcase

        p = build_showcase(a.out)
        console.print(f"[green]showcase[/] {p}")


def _export(a):
    from .export import export_all

    report = export_all(a.weights, a.formats, a.imgsz, workspace_gb=a.workspace,
                        calib_video=a.calib_video, calib_dir=a.calib_dir,
                        calib_images=a.calib_images, calib_algo=a.calib_algo,
                        protect_head=not a.full_int8, verbose=a.verbose)
    t = Table(title="model export")
    cols = [("onnx", "ONNX"), ("onnx-int8", "ONNX INT8"), ("fp32", "TensorRT FP32"),
            ("fp16", "TensorRT FP16"), ("int8", "TensorRT INT8")]
    cols = [c for c in cols if c[0] in a.formats]
    for c in ["model", ".pt MB"] + [c[1] for c in cols]:
        t.add_column(c)

    def cell(d):
        if d is None:
            return "[dim]skipped[/]"
        if "unavailable" in d:
            return "[yellow]not built[/] no TensorRT here"
        if "error" in d:
            return f"[red]failed[/] {d['error'][:60]}"
        return f"[green]{d['mb']} MB[/] ({d['seconds']}s)"

    for r in report:
        t.add_row(r["model"], str(r["pt_mb"]), *[cell(r.get(k)) for k, _ in cols])
    console.print(t)
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(report, indent=2))
    if any("unavailable" in v for r in report for v in r.values() if isinstance(v, dict)):
        console.print("[yellow]TensorRT engines were not built:[/] TensorRT is not installed on "
                      "this machine. Run the same command on the Jetson.")
    if any("error" in v for r in report for v in r.values() if isinstance(v, dict)):
        sys.exit(1)


def _benchmark(a):
    from .benchmark import benchmark, to_markdown

    res = benchmark(a.models, a.video, a.imgsz, a.frames, a.warmup, a.device, a.reference, a.out,
                    a.label)
    console.print(to_markdown(res), markup=False, soft_wrap=True)
    if a.out:
        console.print(f"[green]saved[/] {Path(a.out).with_suffix('.md')} and .json")


def _compare(a):
    from .compare import compare

    compare(a.inputs, a.labels, a.out)
    console.print(f"[green]saved[/] {a.out}")


def _render(a):
    from .render import render

    p = render(a.run, a.out, a.video, a.config, a.zones, a.reapply_rules, not a.no_hud)
    console.print(f"[green]rendered[/] {p}")
    if a.showcase:
        from .showcase import build_showcase

        console.print(f"[green]showcase[/] {build_showcase(str(p.parent))}")


def _showcase(a):
    from .showcase import build_showcase

    p = build_showcase(a.run, a.out, a.video, a.title, not a.no_gif)
    console.print(f"[green]showcase[/] {p}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mantrap", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"mantrap-guard {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("annotate", help="annotate mantrap area, entry line, exit line")
    s.add_argument("--video", required=True)
    s.add_argument("--out", default="configs/zones/zones.json")
    s.add_argument("--frame", type=int, default=0, help="frame index to annotate on")
    s.add_argument("--preview", help="save an image of the annotation")
    s.add_argument("--mantrap", help='headless: polygon as "x,y x,y x,y ..."')
    s.add_argument("--entry-line", help='headless: "x,y x,y"')
    s.add_argument("--exit-line", help='headless: "x,y x,y"')
    s.set_defaults(fn=_annotate)

    s = sub.add_parser("run", help="run the pipeline on a video")
    s.add_argument("--video", required=True)
    s.add_argument("--zones", required=True)
    s.add_argument("--config", default=None, help="YAML overriding the built-in defaults")
    s.add_argument("--model", help="pose model: .pt, .onnx or .engine")
    s.add_argument("--objects-model", help="optional COCO detector for left-behind objects")
    s.add_argument("--imgsz", type=int)
    s.add_argument("--device", help="cpu, 0, ...")
    s.add_argument("--half", action="store_true", help="FP16 inference for .pt on GPU")
    s.add_argument("--flow", choices=["dis", "farneback"])
    s.add_argument("--out", default="outputs/run")
    s.add_argument("--start", type=int, default=0, help="first frame")
    s.add_argument("--max-frames", type=int)
    s.add_argument("--show", action="store_true", help="live window")
    s.add_argument("--dashboard", action="store_true", help="live terminal dashboard")
    s.add_argument("--showcase", action="store_true", help="also build the showcase video")
    s.set_defaults(fn=_run)

    s = sub.add_parser("export", help="export models to ONNX / TensorRT FP16 / TensorRT INT8")
    s.add_argument("--weights", nargs="+", required=True, help="one or more .pt files")
    s.add_argument("--formats", nargs="+", default=["onnx", "fp16", "int8"],
                   choices=["onnx", "onnx-int8", "fp32", "fp16", "int8"],
                   help="onnx-int8 = ONNX Runtime INT8 for CPU; fp32/fp16/int8 = TensorRT engines")
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--calib-video", help="INT8: video to sample calibration frames from")
    s.add_argument("--calib-dir", help="INT8: directory of calibration images")
    s.add_argument("--calib-images", type=int, default=300)
    s.add_argument("--calib-algo", choices=["minmax", "entropy"], default="minmax")
    s.add_argument("--full-int8", action="store_true",
                   help="INT8: also quantise the head (faster, usually hurts keypoints)")
    s.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace, GB")
    s.add_argument("--report", default="outputs/export_report.json")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(fn=_export)

    s = sub.add_parser("benchmark", help="benchmark model variants on video frames")
    s.add_argument("--models", nargs="+", required=True)
    s.add_argument("--video", required=True)
    s.add_argument("--imgsz", type=int, default=640)
    s.add_argument("--frames", type=int, default=150)
    s.add_argument("--warmup", type=int, default=15)
    s.add_argument("--device")
    s.add_argument("--reference", help="model the others are compared against (default: the .pt)")
    s.add_argument("--out", default="outputs/benchmark", help="path stem for .md / .json")
    s.add_argument("--label", help='device name for the comparison table, e.g. "Orin NX 16GB"')
    s.set_defaults(fn=_benchmark)

    s = sub.add_parser("compare", help="merge benchmarks from several devices into one table")
    s.add_argument("--inputs", nargs="+", required=True, help="benchmark .json files")
    s.add_argument("--labels", nargs="+", help="column labels (default: the label in each file)")
    s.add_argument("--out", default="docs/comparison.md")
    s.set_defaults(fn=_compare)

    s = sub.add_parser("render", help="redraw a run's video from detections.jsonl (no inference)")
    s.add_argument("--run", required=True, help="output directory of `mantrap run`")
    s.add_argument("--out", help="output directory (default: the run directory)")
    s.add_argument("--video", help="source video (default: the one recorded in the run)")
    s.add_argument("--config", help="YAML: drawing options and, with --reapply-rules, thresholds")
    s.add_argument("--zones", help="zone file to use instead of the recorded zones")
    s.add_argument("--reapply-rules", action="store_true",
                   help="run the rule engine again on the stored tracks and flow")
    s.add_argument("--no-hud", action="store_true", help="hide the FPS / latency box")
    s.add_argument("--showcase", action="store_true", help="also rebuild the showcase video")
    s.set_defaults(fn=_render)

    s = sub.add_parser("showcase", help="compose input | output | system stats | terminal")
    s.add_argument("--run", required=True, help="output directory of `mantrap run`")
    s.add_argument("--out", help="default: <run>/showcase.mp4")
    s.add_argument("--video", help="input video (default: the one recorded in the run)")
    s.add_argument("--title", default="MANTRAP GUARD")
    s.add_argument("--no-gif", action="store_true")
    s.set_defaults(fn=_showcase)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
