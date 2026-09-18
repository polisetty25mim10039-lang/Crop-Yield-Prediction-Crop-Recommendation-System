"""
generate_diagrams.py
---------------------
Generates the System Architecture and Process Flow diagrams as PNG images
using matplotlib (no internet/Mermaid renderer required). Run this once to
produce docs/architecture_diagram.png and docs/workflow_diagram.png for the
project report and README.

Run:
    python docs/generate_diagrams.py
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = os.path.dirname(__file__)

BOX_STYLE = dict(boxstyle="round,pad=0.4", linewidth=1.5)
COLORS = {
    "data": "#dbeafe", "process": "#dcfce7", "model": "#fef9c3",
    "output": "#fde2e2", "cli": "#ede9fe",
}


def box(ax, x, y, w, h, text, color, fontsize=9):
    rect = FancyBboxPatch((x, y), w, h, **BOX_STYLE, facecolor=color, edgecolor="#333333")
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)
    return (x, y, w, h)


def arrow(ax, start, end, label=None):
    a = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15,
                         color="#444444", linewidth=1.3)
    ax.add_patch(a)
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + 0.15, label, ha="center", fontsize=7.5, color="#555555")


def center_right(b):
    x, y, w, h = b
    return (x + w, y + h / 2)


def center_left(b):
    x, y, w, h = b
    return (x, y + h / 2)


def center_bottom(b):
    x, y, w, h = b
    return (x + w / 2, y)


def center_top(b):
    x, y, w, h = b
    return (x + w / 2, y + h)


def generate_architecture_diagram():
    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.5)
    ax.axis("off")
    ax.set_title("System Architecture — Crop Yield Prediction & Recommendation System",
                 fontsize=13, fontweight="bold", pad=15)

    # Recommendation pipeline (top row)
    b_rec_data = box(ax, 0.3, 5.8, 2.3, 1.0, "Recommendation\nDataset (CSV)\nN,P,K,weather->label", COLORS["data"])
    b_rec_load = box(ax, 3.0, 5.8, 2.1, 1.0, "data_loader.py\nload_recommendation_\ndata()", COLORS["process"])
    b_rec_pre = box(ax, 5.5, 5.8, 2.3, 1.0, "preprocess.py\npreprocess_\nrecommendation()", COLORS["process"])
    b_rec_model = box(ax, 8.2, 5.8, 2.3, 1.0, "model.py\nCrop Recommender\n(RF / GB Classifier)", COLORS["model"])
    b_rec_eval = box(ax, 10.9, 5.8, 1.9, 1.0, "evaluate.py\nAccuracy, F1", COLORS["output"])

    arrow(ax, center_right(b_rec_data), center_left(b_rec_load))
    arrow(ax, center_right(b_rec_load), center_left(b_rec_pre))
    arrow(ax, center_right(b_rec_pre), center_left(b_rec_model))
    arrow(ax, center_right(b_rec_model), center_left(b_rec_eval))

    # Production/yield pipeline (second row)
    b_prod_data = box(ax, 0.3, 4.2, 2.3, 1.0, "Production\nDataset (CSV)\nstate,season,area->prod", COLORS["data"])
    b_prod_load = box(ax, 3.0, 4.2, 2.1, 1.0, "data_loader.py\nload_production_\ndata()", COLORS["process"])
    b_prod_pre = box(ax, 5.5, 4.2, 2.3, 1.0, "preprocess.py\npreprocess_\nproduction()", COLORS["process"])
    b_prod_model = box(ax, 8.2, 4.2, 2.3, 1.0, "model.py\nYield Regressor\n(RF / GB Regressor)", COLORS["model"])
    b_prod_eval = box(ax, 10.9, 4.2, 1.9, 1.0, "evaluate.py\nRMSE, MAE, R2", COLORS["output"])

    arrow(ax, center_right(b_prod_data), center_left(b_prod_load))
    arrow(ax, center_right(b_prod_load), center_left(b_prod_pre))
    arrow(ax, center_right(b_prod_pre), center_left(b_prod_model))
    arrow(ax, center_right(b_prod_model), center_left(b_prod_eval))

    # Persisted artifacts (centered above the CLI layer)
    b_artifacts = box(ax, 3.6, 2.9, 5.8, 0.75,
                       "models/  yield_model.pkl, recommend_model.pkl, yield_encoders.pkl, recommend_encoders.pkl",
                       COLORS["model"], fontsize=8)
    arrow(ax, center_bottom(b_rec_model), (b_artifacts[0] + b_artifacts[2] * 0.75, b_artifacts[1] + b_artifacts[3]))
    arrow(ax, center_bottom(b_prod_model), (b_artifacts[0] + b_artifacts[2] * 0.25, b_artifacts[1] + b_artifacts[3]))

    # CLI layer (bottom) - clean left-to-right flow, predict/recommend side by side
    b_user = box(ax, 0.3, 0.85, 1.5, 0.9, "User\n(terminal)", COLORS["cli"])
    b_cli = box(ax, 2.1, 0.85, 1.9, 0.9, "main.py -> cli.py\nargparse +\nutils.validate_*", COLORS["cli"])
    b_predict = box(ax, 4.3, 1.5, 1.85, 0.75, "predict.py\n(yield)", COLORS["process"], fontsize=8.5)
    b_recommend = box(ax, 4.3, 0.55, 1.85, 0.75, "recommend.py\n(crop)", COLORS["process"], fontsize=8.5)
    b_json = box(ax, 6.55, 1.0, 2.1, 0.9, "JSON output\n+ logger.py entry", COLORS["output"])

    arrow(ax, center_right(b_user), center_left(b_cli))
    arrow(ax, (b_cli[0] + b_cli[2], b_cli[1] + b_cli[3] * 0.75), center_left(b_predict))
    arrow(ax, (b_cli[0] + b_cli[2], b_cli[1] + b_cli[3] * 0.25), center_left(b_recommend))
    arrow(ax, center_right(b_predict), (b_json[0], b_json[1] + b_json[3] * 0.7))
    arrow(ax, center_right(b_recommend), (b_json[0], b_json[1] + b_json[3] * 0.3))
    arrow(ax, (b_artifacts[0] + b_artifacts[2] * 0.35, b_artifacts[1]), center_top(b_predict), label="loads")
    arrow(ax, (b_artifacts[0] + b_artifacts[2] * 0.35, b_artifacts[1]), (b_recommend[0] + b_recommend[2], b_recommend[1] + b_recommend[3]), label="loads")

    # Legend
    legend_items = [
        mpatches.Patch(color=COLORS["data"], label="Raw data"),
        mpatches.Patch(color=COLORS["process"], label="Ingestion / preprocessing / inference"),
        mpatches.Patch(color=COLORS["model"], label="Model training / artifacts"),
        mpatches.Patch(color=COLORS["output"], label="Evaluation / output"),
        mpatches.Patch(color=COLORS["cli"], label="CLI layer"),
    ]
    ax.legend(handles=legend_items, loc="lower right", bbox_to_anchor=(1.0, -0.05),
              ncol=1, fontsize=8, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "architecture_diagram.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def generate_workflow_diagram():
    fig, ax = plt.subplots(figsize=(9, 12.5))
    ax.set_xlim(0, 9)
    ax.set_ylim(-1.2, 12)
    ax.axis("off")
    ax.set_title("Process Flow — `python main.py predict-yield ...`",
                 fontsize=12, fontweight="bold", pad=15)

    steps = [
        ("User runs CLI command with\n--state --season --crop --area", COLORS["cli"]),
        ("cli.py parses arguments\n(argparse subcommand: predict-yield)", COLORS["process"]),
        ("load_config('config.yaml')\n+ get_logger(...)", COLORS["process"]),
        ("Load yield_encoders.pkl\n(State/Season/Crop encoders)", COLORS["model"]),
        ("utils.validate_choice / validate_range\non every field", COLORS["process"]),
        ("predict.py: build feature row,\ncompute area_percentile, encode categoricals", COLORS["process"]),
        ("model.py: load yield_model.pkl\nmodel.predict(features)", COLORS["model"]),
        ("Compute predicted_production_tons\n= predicted_yield * area_ha", COLORS["process"]),
        ("Log request+result to logs/app.log", COLORS["output"]),
        ("Print JSON result to stdout", COLORS["output"]),
    ]

    box_h = 0.62
    gap = 0.5
    top = 11.5
    boxes = []
    for i, (text, color) in enumerate(steps):
        y = top - i * (box_h + gap)
        b = box(ax, 1.6, y, 6.1, box_h, text, color, fontsize=8.5)
        boxes.append(b)

    for i in range(len(boxes) - 1):
        arrow(ax, center_bottom(boxes[i]), center_top(boxes[i + 1]))

    # Error branch, off to the side of the validation step
    err_y = boxes[4][1] - 0.1
    err_box = box(ax, 0.0, err_y - 0.55, 1.5, 1.0,
                   "ValidationError\n-> print error,\nexit 1", COLORS["output"], fontsize=7.5)
    arrow(ax, (boxes[4][0], boxes[4][1] + boxes[4][3] / 2), (err_box[0] + err_box[2], err_box[1] + err_box[3] / 2))

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, "workflow_diagram.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    generate_architecture_diagram()
    generate_workflow_diagram()
