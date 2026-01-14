VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

S2P := data/real/sample.s2p
FMIN := 75
FMAX := 110
MCN ?= 2000
SEED ?= 7
DOEN ?= 500

.PHONY: setup sample analyze mc corner doe demo clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

sample:
	$(PY) -c "import pathlib, skrf as rf; pathlib.Path('data/real').mkdir(parents=True, exist_ok=True); rf.data.ring_slot.write_touchstone('data/real/sample'); print('wrote: data/real/sample.s2p')"

analyze:
	mkdir -p outputs/analyze1
	$(PY) src/analyze_s2p.py --s2p $(S2P) --out outputs/analyze1 --fmin $(FMIN) --fmax $(FMAX)

mc:
	mkdir -p outputs/mc1
	$(PY) src/mc_worst_case.py --s2p $(S2P) --out outputs/mc1 --n $(MCN) --seed $(SEED) --fmin $(FMIN) --fmax $(FMAX)

corner:
	mkdir -p outputs/corner1
	$(PY) src/corner_eval.py --s2p $(S2P) --out outputs/corner1 --fmin $(FMIN) --fmax $(FMAX) --mc-report outputs/mc1/mc_report.json

doe:
	mkdir -p outputs/doe1
	$(PY) src/doe_runner.py --s2p $(S2P) --out outputs/doe1 --n $(DOEN) --fmin $(FMIN) --fmax $(FMAX)

demo: analyze mc corner doe
	@echo "OK: outputs/* generated"

clean:
	rm -rf outputs
