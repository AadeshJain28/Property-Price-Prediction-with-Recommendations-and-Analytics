<#
.SYNOPSIS
    Windows/PowerShell equivalent of the Makefile. `make` does not ship with Windows.

.EXAMPLE
    .\tasks.ps1 setup
    .\tasks.ps1 train
    .\tasks.ps1 test

.NOTES
    If you see "running scripts is disabled on this system", either run
        powershell -ExecutionPolicy Bypass -File .\tasks.ps1 <task>
    or allow local scripts for your user once:
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup','etl','audit','train','test','lint','format','app','api','docker','clean','all')]
    [string]$Task = 'all'
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'

function Step([string]$msg) {
    Write-Host ''
    Write-Host ">>> $msg" -ForegroundColor Cyan
}

function Invoke-Setup {
    Step 'Installing dependencies'
    if (-not $env:VIRTUAL_ENV) {
        Write-Host 'No virtualenv is active. Creating .venv ...' -ForegroundColor Yellow
        python -m venv .venv
        Write-Host 'Now run  .\.venv\Scripts\Activate.ps1  and re-run this script.' -ForegroundColor Yellow
        return
    }
    python -m pip install --upgrade pip
    python -m pip install -r requirements-dev.txt
    python -m pip install -e .
}

function Invoke-Etl    { Step 'ETL';      python -m property_price.etl }
function Invoke-Audit  { Step 'Audit';    python -m property_price.audit }
function Invoke-Train  { Step 'Training'; python -m property_price.train }
function Invoke-Test   { Step 'Tests';    python -m pytest --cov=property_price --cov-report=term-missing }
function Invoke-Lint   { Step 'Lint';     python -m ruff check src tests app; python -m black --check src tests app }
function Invoke-Format { Step 'Format';   python -m black src tests app; python -m ruff check --fix src tests app }
function Invoke-App    { Step 'Streamlit -> http://localhost:8501'; python -m streamlit run app/streamlit_app.py }
function Invoke-Api    { Step 'FastAPI  -> http://localhost:8000/docs'; python -m uvicorn app.api.main:app --reload }
function Invoke-Docker { Step 'Docker';   docker compose up --build }

function Invoke-Clean {
    Step 'Clean'
    Get-ChildItem -Path . -Recurse -Directory -Force |
        Where-Object { $_.Name -in @('__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache') } |
        Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

switch ($Task) {
    'setup'  { Invoke-Setup }
    'etl'    { Invoke-Etl }
    'audit'  { Invoke-Audit }
    'train'  { Invoke-Train }
    'test'   { Invoke-Test }
    'lint'   { Invoke-Lint }
    'format' { Invoke-Format }
    'app'    { Invoke-App }
    'api'    { Invoke-Api }
    'docker' { Invoke-Docker }
    'clean'  { Invoke-Clean }
    'all'    { Invoke-Setup; Invoke-Audit; Invoke-Train; Invoke-Test }
}

Write-Host ''
Write-Host "Done: $Task" -ForegroundColor Green
