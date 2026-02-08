VENV   := .venv
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

S2P    := data/real/sample.s2p
FMIN   := 75
FMAX   := 110
MCN    ?= 2000
SEED   ?= 42
DOEN   ?= 500

.PHONY: setup sample analyze mc corner doe demo lint clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

sample:
	$(PY) -c "\
	  import pathlib, skrf as rf; \
	  pathlib.Path('data/real').mkdir(parents=True, exist_ok=True); \
	  rf.data.ring_slot.write_touchstone('data/real/sample'); \
	  print('Wrote: $(S2P)')"

analyze:
	$(PY) src/analyze_s2p.py --s2p $(S2P) --out outputs/analyze --fmin $(FMIN) --fmax $(FMAX)

mc:
	$(PY) src/mc_worst_case.py --s2p $(S2P) --out outputs/mc --n $(MCN) --seed $(SEED) --fmin $(FMIN) --fmax $(FMAX)

corner:
	$(PY) src/corner_eval.py --s2p $(S2P) --out outputs/corner --fmin $(FMIN) --fmax $(FMAX) --mc-report outputs/mc/mc_report.json

doe:
	$(PY) src/doe_runner.py --s2p $(S2P) --out outputs/doe --n $(DOEN) --fmin $(FMIN) --fmax $(FMAX)

demo: analyze mc corner doe
	@echo "✓ Full pipeline complete — see outputs/"

lint:
	$(PY) -m py_compile src/common.py src/analyze_s2p.py src/mc_worst_case.py src/corner_eval.py src/doe_runner.py
	@echo "✓ All files compile OK"

clean:
	rm -rf outputs
