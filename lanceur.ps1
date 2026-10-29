#Requires -Version 5.0
<#
.SYNOPSIS
    Lanceur Stack Kibali AGI V15 - Architecture API Centrale (Cerveau) + Frontend
.DESCRIPTION
    Lance les embeddings, le serveur RAG, l'API Centrale Kibali (qui gère les LLM), et l'interface Gradio.
.PARAMETER Rebuild   Force la recompilation Flutter
.PARAMETER SkipEmb   Ne relance pas le serveur d'embeddings
.PARAMETER Debug     flutter run (hot reload)
.PARAMETER CheckOnly Verifie l'environnement et quitte
#>
param(
    [switch]$Rebuild,
    [switch]$SkipEmb,
    [switch]$Debug,
    [switch]$CheckOnly
)

$ErrorActionPreference = 'SilentlyContinue'
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

# ==========================================
# CONFIGURATION
# ==========================================
$ROOT_APP    = "D:\kibali-IA"
$ROOT_CPT    = "D:\CPT_Analysis_Studio_PORTABLE"

$LLAMA       = "C:\Users\BELIKANON\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
$MODEL_EMB   = "$ROOT_APP\models\bge-m3-Q4_K_M.gguf"

$PYTHON_FLUTTER  = "$ROOT_APP\env_portable\python.exe"
$PYTHONW_FLUTTER = "$ROOT_APP\env_portable\pythonw.exe"
$PYTHON_RAG      = "$ROOT_CPT\environment\python.exe"
$PYTHON_GRADIO   = "$ROOT_APP\kibali-gradio\.venv\Scripts\python.exe"

$FLUTTER_APP     = "$ROOT_APP\KibaliFlutter"
$BRIDGE_SCRIPT   = "$ROOT_APP\kibali_engine_bridge.py"
$RAG_SERVER_FILE = "$ROOT_CPT\rag_server_fast.py"
$GRADIO_APP_FILE = "$ROOT_APP\kibali-gradio\app.py"
# --- NOUVEAU : Script de l'API Centrale ---
$API_SCRIPT      = "$ROOT_APP\kibali_api.py"

$REQUIRED_DIRS = @(
    "$ROOT_APP\generated_apps",
    "$ROOT_APP\outputs",
    "$ROOT_APP\logs",
    "$ROOT_CPT\outputs",
    "$ROOT_CPT\images",
    "$ROOT_CPT\projects"
)

$EMB_HEALTH   = "http://127.0.0.1:8081/health"
$RAG_HEALTH   = "http://127.0.0.1:8765/tool_schemas"
$API_HEALTH   = "http://127.0.0.1:8000/health"
$GRADIO_HEALTH = "http://127.0.0.1:7860"

# ==========================================
# LOGGING
# ==========================================
$LOG_DIR = "$ROOT_APP\logs"
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }
$LOG_FILE = Join-Path $LOG_DIR ("stack_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
Start-Transcript -Path $LOG_FILE -Force | Out-Null

Write-Host ""
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host "  DEMARRAGE DE LA STACK KIBALI AGI (V15 - API)" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host ("  Log : " + $LOG_FILE) -ForegroundColor DarkGray
Write-Host ""

# ==========================================
# 0. PRE-FLIGHT CHECKS
# ==========================================
Write-Host "[0/7] Verification de l'environnement..." -ForegroundColor Cyan

$fatalErrors = @()
$warnings = @()

function Check-File {
    param($path, $label, [bool]$required = $true)
    if (Test-Path $path) {
        Write-Host ("   [OK] " + $label) -ForegroundColor Green
        return $true
    } else {
        if ($required) {
            Write-Host ("   [MANQUANT] " + $label) -ForegroundColor Red
            $fatalErrors += ("Fichier requis introuvable: " + $path)
            return $false
        } else {
            Write-Host ("   [WARN] " + $label + " (optionnel)") -ForegroundColor Yellow
            return $false
        }
    }
}

Write-Host "   --- Binaires & Modeles ---" -ForegroundColor DarkGray
Check-File $LLAMA "llama-server.exe" | Out-Null
Check-File $MODEL_EMB "Modele embeddings (BGE-M3)" | Out-Null

Write-Host "   --- Python Environments ---" -ForegroundColor DarkGray
Check-File $PYTHON_FLUTTER "Python bridge (env_portable)" | Out-Null
Check-File $PYTHON_RAG "Python RAG (environment)" | Out-Null
Check-File $PYTHON_GRADIO "Python Gradio (kibali-gradio\.venv)" | Out-Null

Write-Host "   --- Scripts Applicatifs ---" -ForegroundColor DarkGray
Check-File $BRIDGE_SCRIPT "Bridge Python (kibali_engine_bridge.py)" | Out-Null
Check-File $RAG_SERVER_FILE "RAG server (rag_server_fast.py)" | Out-Null
Check-File $API_SCRIPT "API Centrale (kibali_api.py)" | Out-Null
Check-File $GRADIO_APP_FILE "Interface Gradio (app.py)" | Out-Null
Check-File "$FLUTTER_APP\pubspec.yaml" "Flutter pubspec.yaml" | Out-Null

Write-Host "   --- Dossiers ---" -ForegroundColor DarkGray
foreach ($d in $REQUIRED_DIRS) {
    if (-not (Test-Path $d)) {
        New-Item -ItemType Directory -Path $d -Force | Out-Null
        Write-Host ("   [+] Cree : " + $d) -ForegroundColor Yellow
    } else {
        Write-Host ("   [OK] " + $d) -ForegroundColor Green
    }
}

Write-Host "   --- Ports ---" -ForegroundColor DarkGray
foreach ($port in @(8080, 8081, 7860, 8765, 8000)) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        Write-Host ("   [!] Port " + $port + " occupe (PID " + $conn[0].OwningProcess + ") - sera libere") -ForegroundColor Yellow
    }
}

Write-Host ""
if ($fatalErrors.Count -gt 0) {
    Write-Host "=================================================" -ForegroundColor Red
    Write-Host "  ERREURS BLOQUANTES - Lancement impossible" -ForegroundColor Red
    Write-Host "=================================================" -ForegroundColor Red
    foreach ($err in $fatalErrors) { Write-Host ("  - " + $err) -ForegroundColor Red }
    Write-Host ""
    Stop-Transcript | Out-Null
    Read-Host "Appuyez sur Entree pour quitter..."
    exit 1
}

Write-Host "[OK] Environnement valide." -ForegroundColor Green
Write-Host ""
if ($CheckOnly) {
    Stop-Transcript | Out-Null
    exit 0
}

# ==========================================
# 1. NETTOYAGE
# ==========================================
Write-Host "[1/7] Nettoyage des anciens processus..." -ForegroundColor Red
Stop-Process -Name "llama-server" -Force -ErrorAction SilentlyContinue
Get-Process python | Where-Object { $_.Path -like "*kibali-IA*" -or $_.Path -like "*CPT_Analysis*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Stop-Process -Name "kibali_terminal" -Force -ErrorAction SilentlyContinue
Get-Process dart | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

foreach ($port in @(8080, 8081, 7860, 8765, 8000)) {
    for ($i = 0; $i -lt 10; $i++) {
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if (-not $conn) { break }
        Start-Sleep -Milliseconds 300
    }
}
Write-Host "   [OK] Processus nettoyes, ports liberes." -ForegroundColor Green

# ==========================================
# 2. EMBEDDINGS (Port 8081)
# ==========================================
if ($SkipEmb) {
    Write-Host "[2/7] SKIP Embeddings (-SkipEmb)" -ForegroundColor DarkGray
} else {
    Write-Host "[2/7] Lancement des Embeddings (BGE-M3) sur port 8081..." -ForegroundColor Cyan
    Start-Process -FilePath $LLAMA -ArgumentList @("-m", $MODEL_EMB, "--embedding", "--port", "8081", "-ngl", "99", "-c", "4096") -WindowStyle Minimized

    Write-Host "   [..] Chargement des embeddings en VRAM..." -ForegroundColor Gray
    $embReady = $false
    for ($i = 0; $i -lt 60; $i++) {
        Start-Sleep 1
        try {
            $r2 = Invoke-RestMethod $EMB_HEALTH -TimeoutSec 1
            if ($r2.status -eq "ok") {
                $embReady = $true
                Write-Host ("       Embeddings prets (" + $i + "s)") -ForegroundColor DarkGreen
                break
            }
        } catch {}
    }
    if (-not $embReady) { Write-Host "[ERREUR] Les embeddings n'ont pas demarre." -ForegroundColor Red; exit 1 }
}

# ==========================================
# 3. SERVEUR RAG & OUTILS (Port 8765)
# ==========================================
Write-Host "[3/7] Lancement du Serveur RAG Kibali (Outils)..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON_RAG -ArgumentList @($RAG_SERVER_FILE) -WorkingDirectory $ROOT_CPT -WindowStyle Minimized

Write-Host "   [..] Initialisation du RAG (FAISS + 390 outils, ~10-30s)..." -ForegroundColor Gray
$ragReady = $false
$toolCount = 0
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep 1
    try {
        $r = Invoke-RestMethod $RAG_HEALTH -TimeoutSec 2
        if ($null -ne $r -and $null -ne $r.tools) {
            $ragReady = $true
            if ($r.count) { $toolCount = $r.count } else { $toolCount = $r.tools.Count }
            Write-Host ("       RAG pret (" + $i + "s) - " + $toolCount + " outils charges") -ForegroundColor DarkGreen
            break
        }
    } catch {}
}

if (-not $ragReady) {
    Write-Host ("[ERREUR] Serveur RAG non accessible sur " + $RAG_HEALTH) -ForegroundColor Red
    exit 1
}
Write-Host "   [OK] Serveur RAG operationnel." -ForegroundColor Green

# ==========================================
# 4. API CENTRALE KIBALI (Port 8000) - LE CERVEAU
# ==========================================
Write-Host "[4/7] Lancement de l'API Centrale Kibali (Cerveau & Agent)..." -ForegroundColor Yellow
# L'API va charger le modèle par défaut (Qwen3-8B) et le gérer en tâche de fond
Start-Process -FilePath "cmd.exe" -ArgumentList "/k title Kibali API (Cerveau) && `"$PYTHON_FLUTTER`" kibali_api.py" -WorkingDirectory $ROOT_APP

Write-Host "   [..] Initialisation de l'Agent et chargement du LLM (~15s)..." -ForegroundColor Gray
$apiReady = $false
for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep 1
    try {
        $r = Invoke-RestMethod $API_HEALTH -TimeoutSec 2
        if ($r.status -eq "ok") {
            $apiReady = $true
            Write-Host ("       API Centrale prete ! (Modele actif : " + $r.active_model + ")") -ForegroundColor DarkGreen
            break
        }
    } catch {}
}

if (-not $apiReady) {
    Write-Host "[ERREUR] L'API Centrale n'a pas demarre correctement." -ForegroundColor Red
    exit 1
}

# ==========================================
# 5. INTERFACE GRADIO (Port 7860) - LE FRONTEND
# ==========================================
Write-Host "[5/7] Lancement de l'interface Gradio (Frontend)..." -ForegroundColor Yellow
# Gradio est maintenant un simple client qui appelle l'API
Start-Process -FilePath "cmd.exe" -ArgumentList "/k title Kibali UI && `"$PYTHON_GRADIO`" app.py" -WorkingDirectory "$ROOT_APP\kibali-gradio"

Write-Host "   [..] Attente de l'interface Gradio..." -ForegroundColor Gray
$gradioReady = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep 1
    try {
        $r = Invoke-WebRequest $GRADIO_HEALTH -TimeoutSec 2 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            $gradioReady = $true
            Write-Host "       Interface Gradio prete !" -ForegroundColor DarkGreen
            break
        }
    } catch {}
}
if ($gradioReady) {
    Write-Host "   [OK] Interface Gradio accessible sur http://127.0.0.1:7860" -ForegroundColor Green
} else {
    Write-Host "   [WARN] Gradio met du temps a demarrer." -ForegroundColor Yellow
}

# ==========================================
# 6. FLUTTER
# ==========================================
Write-Host "[6/7] Preparation de Flutter..." -ForegroundColor Yellow
Set-Location $FLUTTER_APP

$buildDir = "$FLUTTER_APP\build\windows\x64\runner\Release"
$exe = $null
if (Test-Path $buildDir) {
    $exe = Get-ChildItem -Path $buildDir -Filter "*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
}

function Test-FlutterNeedsBuild {
    param($FlutterApp, $ExePath)
    if (-not (Test-Path $ExePath)) { return $true }
    $exeTime = (Get-Item $ExePath).LastWriteTime
    $newestDart = Get-ChildItem -Path (Join-Path $FlutterApp "lib") -Recurse -Filter "*.dart" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if ($newestDart -and $newestDart.LastWriteTime -gt $exeTime) { return $true }
    return $false
}

$needsBuild = $false
if ($Rebuild) { $needsBuild = $true }
elseif ($exe) { $needsBuild = Test-FlutterNeedsBuild -FlutterApp $FLUTTER_APP -ExePath $exe.FullName }
else { $needsBuild = $true }

$ExePath = $null

if ($Debug) {
    Write-Host "   [DEBUG] flutter run -d windows..." -ForegroundColor Cyan
    Start-Process -FilePath "cmd.exe" -ArgumentList "/k title Kibali Debug && flutter run -d windows" -WorkingDirectory $FLUTTER_APP
}
elseif ($needsBuild) {
    Write-Host "   [..] Compilation complete (2-5 min)..." -ForegroundColor Gray
    if (Test-Path "$FLUTTER_APP\build") { Remove-Item -Path "$FLUTTER_APP\build" -Recurse -Force -ErrorAction SilentlyContinue }
    
    flutter clean 2>&1 | Out-Null
    flutter pub get 2>&1 | Out-Null
    $buildOutput = (flutter build windows 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] flutter build windows a echoue" -ForegroundColor Red
        Write-Host $buildOutput -ForegroundColor DarkGray
        exit 1
    }
    $exe = Get-ChildItem -Path $buildDir -Filter "*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $exe) { Write-Host "[ERREUR] Aucun .exe trouve." -ForegroundColor Red; exit 1 }
    $ExePath = $exe.FullName
    Write-Host "   [OK] Compilation terminee." -ForegroundColor Green
} else {
    $ExePath = $exe.FullName
    Write-Host "   [OK] Exe Flutter a jour." -ForegroundColor Green
}

# ==========================================
# 7. LANCEMENT FLUTTER & STATUS
# ==========================================
if (-not $Debug -and $ExePath -and (Test-Path $ExePath)) {
    Write-Host "[7/7] Lancement de l'application Flutter..." -ForegroundColor Yellow
    Start-Process -FilePath $ExePath -WorkingDirectory (Split-Path $ExePath)
    Write-Host ("   [OK] Flutter lance : " + (Split-Path $ExePath -Leaf)) -ForegroundColor Green
} else {
    Write-Host "   [INFO] Flutter non lance (mode debug ou erreur)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host "  STACK KIBALI AGI V15 (API CENTRALE) OPERATIONNELLE" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host "  API Centrale (Cerveau)   : http://127.0.0.1:8000" -ForegroundColor Cyan
Write-Host "  Gradio UI (Frontend)     : http://127.0.0.1:7860" -ForegroundColor Cyan
Write-Host "  Embeddings Server        : http://127.0.0.1:8081" -ForegroundColor Cyan
Write-Host "  RAG Server (Outils)      : http://127.0.0.1:8765" -ForegroundColor Cyan
Write-Host ("  Apps dir                 : " + $ROOT_APP + "\generated_apps") -ForegroundColor Cyan
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "Ctrl+C pour arreter la surveillance (services restent actifs)." -ForegroundColor Yellow
Write-Host ""

while ($true) {
    Start-Sleep -Seconds 5
    $servicesDown = @()
    try { $null = Invoke-RestMethod $EMB_HEALTH -TimeoutSec 2 } catch { $servicesDown += "Embeddings" }
    try { $null = Invoke-RestMethod $RAG_HEALTH -TimeoutSec 2 } catch { $servicesDown += "RAG" }
    try { $null = Invoke-RestMethod $API_HEALTH -TimeoutSec 2 } catch { $servicesDown += "API Centrale" }
    try { $null = Invoke-WebRequest $GRADIO_HEALTH -UseBasicParsing -TimeoutSec 2 } catch { $servicesDown += "Gradio UI" }
    
    if ($servicesDown.Count -gt 0) {
        $stamp = Get-Date -Format 'HH:mm:ss'
        $joined = $servicesDown -join ', '
        Write-Host ("[" + $stamp + "] [!] Services DOWN: " + $joined) -ForegroundColor Yellow
    }
}

Stop-Transcript | Out-Null