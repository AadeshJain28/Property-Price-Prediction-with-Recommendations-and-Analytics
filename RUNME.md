# Run this first

The repo is complete, but the **model artefacts and model metrics do not exist yet**. They
were not generated in the environment this was built in, and no number has been written
anywhere that was not actually measured.

## Windows / PowerShell

`make` is a Unix tool and does not ship with Windows. Use `tasks.ps1`, which has the same
targets:

```powershell
.\tasks.ps1 setup     # pip install into the active venv
.\tasks.ps1 audit     # reproduces reports/data_audit.json
.\tasks.ps1 train     # generates every model number
.\tasks.ps1 test
.\tasks.ps1 api       # http://localhost:8000/docs
.\tasks.ps1 app       # http://localhost:8501
```

If PowerShell blocks the script:

```powershell
powershell -ExecutionPolicy Bypass -File .\tasks.ps1 train
# or, once, for your user:
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### Or skip the wrapper entirely

With your venv active, these are the only commands that matter:

```powershell
python -m pip install -r requirements-dev.txt
python -m pip install -e .
$env:PYTHONPATH = "$PWD\src"
python -m property_price.audit
python -m property_price.train
python -m pytest
```

## macOS / Linux

```bash
make setup && make audit && make train && make test
```

## Before you quote anything

The **audit figures are already measured** and safe to use — they reproduce every time you
run `audit`, and CI regenerates them. The **model figures are not**, until `train` has run
on your machine and written `reports/summary.json`.
