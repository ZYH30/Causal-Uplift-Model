# Reproducible figure generation

## Run

From the repository root:

```bash
python figure_generation/scripts/plot_all_figures.py
```

Individual figures can be regenerated with `plot_figure_2.py` through `plot_figure_5.py`. Use `--output-dir PATH` to change the output directory.

The MIMIC-specific frozen inputs can be rebuilt from the committed aggregate
results before plotting:

```bash
python figure_generation/scripts/prepare_mimic_figure_data.py
```

## Inputs

All plotting inputs are frozen aggregate CSV/JSON files in `data/`. The scripts do not fit models, rerun experiments, access clinical row-level data, or alter estimates. Derived visual quantities, such as paired differences and the treatment-budget difference strip, are deterministic transformations of those files.

## Outputs

Each script writes:

- vector PDF;
- vector SVG;
- 600-dpi PNG.

The shared `scripts/figure_style.py` defines the fixed random seed, colorblind-safe palette, sans-serif typography, line and marker scales, panel labels, and export settings.

## Figure-to-data map

- Figure 2: `systematic_graph_summary.csv`, `systematic_cate_summary.csv`, and `tlcd_component_ablation_summary.csv`.
- Figure 3: cohort flow, era summary, temporal adjustment-set balance,
  post-discovery split summary, and paired temporal comparisons.
- Figure 4: temporal benefit and bedside profiles, hemoglobin response, and
  treatment-budget policy curves.
- Figure 5: graph-audit summary, edge-decision ledger, and API-usage summary.
