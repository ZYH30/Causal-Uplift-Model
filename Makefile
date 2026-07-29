PYTHON ?= .venv/bin/python
MPLCONFIGDIR ?= /tmp/matplotlib-ccia-ocu
export MPLCONFIGDIR

.PHONY: env test synthetic direction ccia-synthetic llm-dryrun llm-api llm-variants nhanes-confirmatory nhanes-survey figures

env:
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

test:
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py' -v

synthetic:
	$(PYTHON) experiments/reproduce_synthetic.py --n 10000 --seed 2026 --n-trials 16 --out-dir results/generated/synthetic

direction:
	$(PYTHON) experiments/benchmark_direction_multiseed.py --n 6000 --seeds 2026,2027,2028,2029,2030 --methods pair_spearman_anm task_adaptive --max-context 3 --max-hsic-samples 1000 --lgb-estimators 80 --gain-estimators 50 --out-dir results/generated/direction

ccia-synthetic:
	$(PYTHON) experiments/run_ccia_synthetic.py --n 10000 --include-colliders --no-uplift-temporal-order --direction-strategy task_adaptive --max-neighbors 16 --max-corr-neighbors 10 --max-importance-neighbors 12 --max-condition-candidates 10 --max-condition-set-size 2 --max-search-depth 1 --ci-theta 0.05 --ccia-lgb-estimators 80 --out-dir results/generated/ccia-synthetic

llm-dryrun:
	$(PYTHON) experiments/run_nhanes_llm_dryrun.py

llm-api:
	@test "$(ALLOW_PAID_API)" = "1" || (echo "Paid API calls are locked. Review the dry-run budget and set ALLOW_PAID_API=1." && exit 1)
	ALLOW_PAID_API=1 $(PYTHON) experiments/run_nhanes_llm_adjudication.py

llm-variants:
	$(PYTHON) experiments/build_nhanes_llm_graph_variants.py

nhanes-confirmatory:
	$(PYTHON) empirical_experiments/nhanes_fixed_confirmatory.py --cache-dir data/nhanes_raw --out-dir results/generated/nhanes-confirmatory --seed-start 2040 --repeats 20 --bootstrap 10000

nhanes-survey:
	$(PYTHON) empirical_experiments/nhanes_survey_aipw_sensitivity.py --cache-dir data/nhanes_raw --out-dir results/generated/nhanes-survey

figures:
	$(PYTHON) empirical_experiments/build_final_manuscript_figures.py --output-dir figures
