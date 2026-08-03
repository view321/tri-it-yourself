.PHONY: test smoke sizes ablate main baselines clean

PY ?= python

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m pytest -q
	$(PY) -m tri.train --preset smoke --quant sign
	$(PY) -m tri.train --preset smoke --quant ste
	$(PY) -m tri.train --preset smoke --quant bf16

sizes:
	$(PY) -c "from tri.config import build_configs, PRESETS; \
	[print(p, build_configs(p)[0].param_counts()) for p in PRESETS]"

DATA ?= --dataset bin --data-dir data

ablate:
	$(PY) -m tri.ablate --study sign  --trials 40 --preset tiny --steps 800 $(DATA)
	$(PY) -m tri.ablate --study modes --trials 24 --preset tiny --steps 800 $(DATA)
	$(PY) -m tri.ablate --study loops --trials 8  --preset tiny --steps 800 $(DATA)

main:
	$(PY) -m tri.train --preset main --quant sign --dataset bin --data-dir data --ckpt-every 1000

baselines:
	$(PY) -m tri.train --preset main --quant bf16 --dataset bin --data-dir data --steps 3000
	$(PY) -m tri.train --preset main --quant ste  --dataset bin --data-dir data --steps 3000

clean:
	rm -rf runs .pytest_cache **/__pycache__
