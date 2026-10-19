# Manuscript result figures

This directory generates the final Figures 3-6 and their 19 individual panels
from the included aggregate and seed-level results. It can run independently
of the experiment code. Python 3.11 or newer is required.

## Install

From the repository root:

```bash
python3 -m venv figure_generation/.venv
figure_generation/.venv/bin/python -m pip install -r figure_generation/requirements.txt
```

The pinned requirements reproduce the plotting environment. Matplotlib
provides the DejaVu Sans font used by the figures. Rendering uses the
noninteractive Agg backend.

## Generate figures

Generate all four result figures:

```bash
figure_generation/.venv/bin/python figure_generation/generate.py
```

Generate one figure, export individual panels, or select a destination:

```bash
figure_generation/.venv/bin/python figure_generation/generate.py --figures 5
figure_generation/.venv/bin/python figure_generation/generate.py --panels
figure_generation/.venv/bin/python figure_generation/generate.py \
  --figures 3 4 --output-dir /path/to/output --dpi 1200
```

Paths to the default inputs and outputs are resolved relative to the code,
so the command also works from a different working directory. An existing
environment with these dependencies can be used instead of the local one.
The repository's `make figures` target calls the same entry point; use
`make figures PYTHON=figure_generation/.venv/bin/python` for this environment.

## Outputs

The default output directory is `../figures/` relative to this directory.
Every result figure is exported as vector PDF, editable-text SVG and 900-dpi
PNG. `--dpi` changes PNG resolution and must exceed 600. `--panels` additionally
writes PDF, SVG and PNG for all requested panels to `figures/panels/`.

| Figure | Output stem | Physical size | Panels | Plotting module |
|---|---|---|---|---|
| 3 | `Figure_3_MIMIC` | 162 x 109 mm | a-e | `scripts/plot_figure_3.py` |
| 4 | `Figure_4_Clinical_profiles` | 162 x 112 mm | a-d | `scripts/plot_figure_4.py` |
| 5 | `Figure_5_Role_mechanism` | 162 x 150 mm | a-f | `scripts/plot_figure_5.py` |
| 6 | `Figure_6_NHANES` | 162 x 132 mm | a-d | `scripts/plot_figure_6.py` |

The conceptual artwork for Figures 1-2 is supplied in the repository's
`figures/` directory. The plotting entry point produces Figures 3-6.

## Inputs and statistical conventions

`data/mimic/` supplies the transfusion comparisons, covariate balance, clinical
eras, patient profiles, hemoglobin response and treatment-capacity curves.
`data/synthetic/` supplies the known-role comparison, graph recovery,
component ablation and matched downstream effect estimates.
`data/nhanes/` supplies paired medication-study gains, their reported mean
intervals, cohort metadata, and the graphs and role assignments used by the
variable-flow panels.

The programs calculate paired differences, unit conversions, displayed
quantiles and the within-row standardized clinical profiles. Reported
confidence intervals are read from the supplied results. They do not train
models, repeat experiments, bootstrap data, access patient-level records or
make API calls.

Fig. 3 preserves split identities across all three gain panels. Fig. 5 uses
standard errors for recovery means, standard deviations for component
comparisons, and all matched seed values for downstream gaps; values outside
the local axes remain explicitly annotated. Fig. 6 retains all 20 paired
splits per task and distinguishes final relation decisions from earlier
retention decisions. Historical experiment identifiers in the input files
remain stable; the displayed method names are TLCD and RITE.
