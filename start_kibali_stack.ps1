#Requires -Version 5.0
<#
.SYNOPSIS
    Lanceur Stack Kibali AGI V13.2 (ASCII-safe, import-test robuste)
.PARAMETER Rebuild   Force la recompilation Flutter
.PARAMETER SkipLLM   Ne relance pas LLM/Embeddings
.PARAMETER Debug     flutter run (hot reload)
.PARAMETER CheckOnly Verifie l'environnement et quitte
#>
param(
    [switch]$Rebuild,
    [switch]$SkipLLM,
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
$ROOT_MODELS = "$ROOT_CPT\models"

$LLAMA       = "C:\Users\BELIKANON\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
$MODEL_LLM   = "$ROOT_MODELS\Qwen3-8B-Q4_K_M.gguf"
$MODEL_EMB   = "$ROOT_MODELS\bge-m3-Q4_K_M.gguf"

$PYTHON_FLUTTER  = "$ROOT_APP\env_portable\python.exe"
$PYTHONW_FLUTTER = "$ROOT_APP\env_portable\pythonw.exe"
$PYTHON_RAG      = "$ROOT_CPT\environment\python.exe"

$FLUTTER_APP     = "$ROOT_APP\KibaliFlutter"
$BRIDGE_SCRIPT   = "$ROOT_APP\kibali_engine_bridge.py"
$RAG_SERVER_FILE = "$ROOT_CPT\rag_server_fast.py"

$REQUIRED_DIRS = @(
    "$ROOT_APP\generated_apps",
    "$ROOT_APP\outputs",
    "$ROOT_APP\logs",
    "$ROOT_CPT\outputs",
    "$ROOT_CPT\images",
    "$ROOT_CPT\projects"
)

$LLM_HEALTH = "http://127.0.0.1:8080/health"
$EMB_HEALTH = "http://127.0.0.1:8081/health"
$RAG_HEALTH = "http://127.0.0.1:8765/tool_schemas"

# ==========================================
# LOGGING
# ==========================================
$LOG_DIR = "$ROOT_APP\logs"
if (-not (Test-Path $LOG_DIR)) { New-Item -ItemType Directory -Path $LOG_DIR -Force | Out-Null }
$LOG_FILE = Join-Path $LOG_DIR ("stack_" + (Get-Date -Format 'yyyyMMdd_HHmmss') + ".log")
Start-Transcript -Path $LOG_FILE -Force | Out-Null

Write-Host ""
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host "  DEMARRAGE DE LA STACK KIBALI AGI (V13.2)" -ForegroundColor Green
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
            Write-Host ("              -> " + $path) -ForegroundColor DarkGray
            return $false
        } else {
            Write-Host ("   [WARN] " + $label + " (optionnel)") -ForegroundColor Yellow
            return $false
        }
    }
}

Write-Host "   --- Binaires ---" -ForegroundColor DarkGray
if (-not (Check-File $LLAMA "llama-server.exe")) { $fatalErrors += ("llama-server.exe introuvable: " + $LLAMA) }
if (-not (Check-File $MODEL_LLM "Modele LLM Qwen3-8B")) { $fatalErrors += ("Modele LLM manquant: " + $MODEL_LLM) }
if (-not (Check-File $MODEL_EMB "Modele embeddings BGE-M3")) { $fatalErrors += ("Modele embeddings manquant: " + $MODEL_EMB) }

Write-Host "   --- Python ---" -ForegroundColor DarkGray
if (-not (Check-File $PYTHON_FLUTTER "Python bridge (env_portable)")) { $fatalErrors += ("Python env_portable introuvable: " + $PYTHON_FLUTTER) }
if (-not (Check-File $PYTHONW_FLUTTER "PythonW (lancement apps PyQt6)")) {
    $warnings += "pythonw.exe manquant - apps PyQt6 afficheront une console"
}
if (-not (Check-File $PYTHON_RAG "Python RAG (environment)")) { $fatalErrors += ("Python RAG introuvable: " + $PYTHON_RAG) }

Write-Host "   --- Scripts ---" -ForegroundColor DarkGray
if (-not (Check-File $BRIDGE_SCRIPT "Bridge Python (kibali_engine_bridge.py)")) { $fatalErrors += ("Bridge manquant: " + $BRIDGE_SCRIPT) }
if (-not (Check-File $RAG_SERVER_FILE "rag_server_fast.py")) { $fatalErrors += ("RAG server manquant: " + $RAG_SERVER_FILE) }

Write-Host "   --- Flutter ---" -ForegroundColor DarkGray
if (-not (Check-File "$FLUTTER_APP\pubspec.yaml" "Flutter pubspec.yaml")) { $fatalErrors += ("Flutter app introuvable: " + $FLUTTER_APP) }

# ---- Dépendances Python (dry-run import) ----
if (Test-Path $PYTHON_FLUTTER) {
    Write-Host "   --- Dependances LangChain ---" -ForegroundColor DarkGray
    $importCmd = "from langchain_openai import ChatOpenAI; from langchain_core.prompts import PromptTemplate; from langchain_classic.agents import create_react_agent, AgentExecutor, Tool; print('OK')"
    $importCheck = (& $PYTHON_FLUTTER -c $importCmd 2>&1 | Out-String)
    if ($importCheck -notmatch 'OK') {
        Write-Host "   [ECHEC] Dependances LangChain manquantes dans env_portable" -ForegroundColor Red
        Write-Host $importCheck.Trim() -ForegroundColor DarkGray
        $fatalErrors += ("Execute : " + $PYTHON_FLUTTER + " -m pip install langchain-openai langchain-classic langchain-core")
    } else {
        Write-Host "   [OK] langchain_openai + langchain_classic" -ForegroundColor Green
    }

    # ---- Module outils : test d'import RÉEL (pas de check fichier) ----
    Write-Host "   --- Module outils (from outils import ToolManager) ---" -ForegroundColor DarkGray
    $importOutilsCmd = "import sys; sys.path.insert(0, r'D:\kibali-IA'); from outils import ToolManager; print('OK:' + ToolManager.__module__)"
    $outilsImport = (& $PYTHON_FLUTTER -c $importOutilsCmd 2>&1 | Out-String)
    if ($outilsImport -match 'OK:') {
        $modLine = ($outilsImport -split "`n" | Where-Object { $_ -match 'OK:' } | Select-Object -First 1).Trim()
        Write-Host ("   [OK] " + $modLine) -ForegroundColor Green
    } else {
        Write-Host "   [ECHEC] from outils import ToolManager" -ForegroundColor Red
        Write-Host $outilsImport.Trim() -ForegroundColor DarkGray

        Write-Host "   Diagnostic : recherche de outils.py/outils/" -ForegroundColor Yellow
        $found = $false
        foreach ($c in @(
            "D:\kibali-IA\outils.py",
            "D:\kibali-IA\outils\__init__.py",
            "D:\CPT_Analysis_Studio_PORTABLE\outils.py",
            "D:\CPT_Analysis_Studio_PORTABLE\outils\__init__.py"
        )) {
            if (Test-Path $c) {
                Write-Host ("     TROUVE : " + $c) -ForegroundColor Green
                $found = $true
            }
        }
        if (-not $found) {
            $rec = Get-ChildItem -Path "D:\kibali-IA" -Recurse -Filter "outils.py" -ErrorAction SilentlyContinue | Select-Object -First 3
            foreach ($f in $rec) {
                Write-Host ("     TROUVE (recherche) : " + $f.FullName) -ForegroundColor Green
                $found = $true
            }
            $recPkg = Get-ChildItem -Path "D:\kibali-IA" -Recurse -Filter "__init__.py" -ErrorAction SilentlyContinue |
                      Where-Object { $_.Directory.Name -eq "outils" } | Select-Object -First 3
            foreach ($f in $recPkg) {
                Write-Host ("     TROUVE (package) : " + $f.FullName) -ForegroundColor Green
                $found = $true
            }
        }
        if ($found) {
            Write-Host "   ACTION : copie outils.py (ou le dossier outils/) vers D:\kibali-IA\" -ForegroundColor Yellow
        } else {
            Write-Host "   ACTION : cree D:\kibali-IA\outils.py (avec class ToolManager)" -ForegroundColor Yellow
        }
        $fatalErrors += "from outils import ToolManager echoue (outils.py introuvable ou import casse)"
    }
}

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
foreach ($port in @(8080, 8081, 8765)) {
    $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    if ($conn) {
        Write-Host ("   [!] Port " + $port + " deja occupe (PID " + $conn[0].OwningProcess + ") - sera libere") -ForegroundColor Yellow
    } else {
        Write-Host ("   [OK] Port " + $port + " libre") -ForegroundColor Green
    }
}

Write-Host ""
if ($fatalErrors.Count -gt 0) {
    Write-Host "=================================================" -ForegroundColor Red
    Write-Host "  ERREURS BLOQUANTES - Lancement impossible" -ForegroundColor Red
    Write-Host "=================================================" -ForegroundColor Red
    $i = 1
    foreach ($err in $fatalErrors) {
        Write-Host ("  " + $i + ". " + $err) -ForegroundColor Red
        $i++
    }
    Write-Host ""
    Stop-Transcript | Out-Null
    Read-Host "Appuyez sur Entree pour quitter..."
    exit 1
}

if ($warnings.Count -gt 0) {
    Write-Host "[AVERTISSEMENTS]" -ForegroundColor Yellow
    foreach ($w in $warnings) { Write-Host ("  - " + $w) -ForegroundColor Yellow }
    Write-Host ""
}

Write-Host "[OK] Environnement valide." -ForegroundColor Green
Write-Host ""

if ($CheckOnly) {
    Write-Host "Mode -CheckOnly : sortie sans lancement." -ForegroundColor Cyan
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

foreach ($port in @(8080, 8081, 8765)) {
    for ($i = 0; $i -lt 10; $i++) {
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if (-not $conn) { break }
        Start-Sleep -Milliseconds 300
    }
}
Write-Host "   [OK] Processus nettoyes, ports liberes." -ForegroundColor Green

# ==========================================
# 2. LLM & EMBEDDINGS
# ==========================================
if ($SkipLLM) {
    Write-Host "[2/7] SKIP LLM/Embeddings (-SkipLLM)" -ForegroundColor DarkGray
} else {
    Write-Host "[2/7] Lancement du LLM (Qwen3-8B) et Embeddings (BGE-M3)..." -ForegroundColor Cyan
    Start-Process -FilePath $LLAMA -ArgumentList @("-m", $MODEL_LLM, "--jinja", "--port", "8080", "-ngl", "99", "-c", "32768", "-fa", "on") -WindowStyle Minimized
    Start-Process -FilePath $LLAMA -ArgumentList @("-m", $MODEL_EMB, "--embedding", "--port", "8081", "-ngl", "99", "-c", "4096") -WindowStyle Minimized

    Write-Host "   [..] Chargement des modeles en VRAM (Patientez ~20s)..." -ForegroundColor Gray
    $llmReady = $false
    $embReady = $false
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep 1
        if (-not $llmReady) {
            try {
                $r1 = Invoke-RestMethod $LLM_HEALTH -TimeoutSec 1
                if ($r1.status -eq "ok") {
                    $llmReady = $true
                    Write-Host ("       LLM pret (" + $i + "s)") -ForegroundColor DarkGreen
                }
            } catch {}
        }
        if (-not $embReady) {
            try {
                $r2 = Invoke-RestMethod $EMB_HEALTH -TimeoutSec 1
                if ($r2.status -eq "ok") {
                    $embReady = $true
                    Write-Host ("       Embeddings prets (" + $i + "s)") -ForegroundColor DarkGreen
                }
            } catch {}
        }
        if ($llmReady -and $embReady) { break }
    }

    if (-not $llmReady -or -not $embReady) {
        Write-Host "[ERREUR] Les serveurs llama n'ont pas demarre dans les delais." -ForegroundColor Red
        Write-Host ("         LLM : " + $llmReady) -ForegroundColor Red
        Write-Host ("         EMB : " + $embReady) -ForegroundColor Red
        Stop-Transcript | Out-Null
        Read-Host "Appuyez sur Entree pour quitter..."
        exit 1
    }
    Write-Host "   [OK] Modeles LLM + Embeddings charges !" -ForegroundColor Green
}

# ==========================================
# 3. SERVEUR RAG
# ==========================================
Write-Host "[3/7] Lancement du Serveur RAG Kibali (rag_server_fast.py)..." -ForegroundColor Yellow
Start-Process -FilePath $PYTHON_RAG -ArgumentList @($RAG_SERVER_FILE) -WorkingDirectory $ROOT_CPT -WindowStyle Minimized

Write-Host "   [..] Initialisation du RAG (FAISS + outils, ~10-30s)..." -ForegroundColor Gray
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
    Stop-Transcript | Out-Null
    Read-Host "Appuyez sur Entree pour quitter..."
    exit 1
}
Write-Host "   [OK] Serveur RAG operationnel." -ForegroundColor Green

# ==========================================
# 4. FLUTTER
# ==========================================
Write-Host "[4/7] Preparation de Flutter..." -ForegroundColor Yellow
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
    if ($newestDart -and $newestDart.LastWriteTime -gt $exeTime) {
        Write-Host ("   [BUILD] " + $newestDart.Name + " plus recent que l'exe") -ForegroundColor DarkYellow
        return $true
    }
    $pubspecPath = Join-Path $FlutterApp "pubspec.yaml"
    if (Test-Path $pubspecPath) {
        $pubspec = Get-Item $pubspecPath
        if ($pubspec.LastWriteTime -gt $exeTime) {
            Write-Host "   [BUILD] pubspec.yaml plus recent que l'exe" -ForegroundColor DarkYellow
            return $true
        }
    }
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
    if (Test-Path "$FLUTTER_APP\windows\flutter\ephemeral") { Remove-Item -Path "$FLUTTER_APP\windows\flutter\ephemeral" -Recurse -Force -ErrorAction SilentlyContinue }

    flutter clean 2>&1 | Out-Null
    flutter pub get 2>&1 | Out-Null
    $buildOutput = (flutter build windows 2>&1 | Out-String)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERREUR] flutter build windows a echoue" -ForegroundColor Red
        Write-Host $buildOutput -ForegroundColor DarkGray
        Stop-Transcript | Out-Null
        Read-Host "Appuyez sur Entree pour quitter..."
        exit 1
    }
    $exe = Get-ChildItem -Path $buildDir -Filter "*.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $exe) {
        Write-Host ("[ERREUR] Aucun .exe dans " + $buildDir) -ForegroundColor Red
        Stop-Transcript | Out-Null
        Read-Host "Appuyez sur Entree pour quitter..."
        exit 1
    }
    $ExePath = $exe.FullName
    Write-Host "   [OK] Compilation terminee." -ForegroundColor Green
} else {
    $ExePath = $exe.FullName
    Write-Host "   [OK] Exe a jour - compilation ignoree." -ForegroundColor Green
    Write-Host "        (Utilise -Rebuild pour forcer)" -ForegroundColor DarkGray
}

# ==========================================
# 5. LANCEMENT FLUTTER
# ==========================================
if (-not $Debug -and $ExePath -and (Test-Path $ExePath)) {
    Write-Host "[5/7] Lancement de l'application Flutter..." -ForegroundColor Yellow
    Start-Process -FilePath $ExePath -WorkingDirectory (Split-Path $ExePath)
    Write-Host ("   [OK] Flutter lance : " + (Split-Path $ExePath -Leaf)) -ForegroundColor Green
} elseif (-not $Debug) {
    Write-Host "[ERREUR] Exe Flutter introuvable." -ForegroundColor Red
    Stop-Transcript | Out-Null
    Read-Host "Appuyez sur Entree pour quitter..."
    exit 1
} else {
    Write-Host "[5/7] Flutter en mode debug." -ForegroundColor Cyan
}

# ==========================================
# 6. VERIFICATION FINALE
# ==========================================
Write-Host "[6/7] Verification finale des services..." -ForegroundColor Yellow
Start-Sleep -Seconds 2

$services = @(
    @{ Name = "LLM Server"; URL = "http://127.0.0.1:8080/health";       Tag = "[LLM]" },
    @{ Name = "Embeddings"; URL = "http://127.0.0.1:8081/health";       Tag = "[EMB]" },
    @{ Name = "RAG Server"; URL = "http://127.0.0.1:8765/tool_schemas"; Tag = "[RAG]" }
)
foreach ($svc in $services) {
    try {
        $null = Invoke-RestMethod $svc.URL -TimeoutSec 2
        Write-Host ("   " + $svc.Tag + " " + $svc.Name + " : OK") -ForegroundColor Green
    } catch {
        Write-Host ("   " + $svc.Tag + " " + $svc.Name + " : INDISPONIBLE") -ForegroundColor Red
    }
}

Write-Host "   [..] Attente demarrage bridge..." -ForegroundColor Gray
Start-Sleep -Seconds 3
$bridgeProc = Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "*env_portable*" }
if ($bridgeProc) {
    Write-Host ("   [BRIDGE] Python : OK (PID " + $bridgeProc[0].Id + ")") -ForegroundColor Green
} else {
    Write-Host "   [BRIDGE] Python : En attente (demarre avec Flutter)" -ForegroundColor Yellow
}

# ==========================================
# 7. STATUS FINAL
# ==========================================
Write-Host "[7/7] Stack operationnelle." -ForegroundColor Green
Write-Host ""
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host "  STACK KIBALI AGI V13.2 OPERATIONNELLE" -ForegroundColor Green
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host "  LLM Server   : http://127.0.0.1:8080" -ForegroundColor Cyan
Write-Host "  Embeddings   : http://127.0.0.1:8081" -ForegroundColor Cyan
Write-Host "  RAG Server   : http://127.0.0.1:8765" -ForegroundColor Cyan
Write-Host ("  Bridge       : " + $BRIDGE_SCRIPT) -ForegroundColor Cyan
Write-Host "  Flutter App  : Lancee en natif" -ForegroundColor Cyan
Write-Host ("  Apps dir     : " + $ROOT_APP + "\generated_apps") -ForegroundColor Cyan
Write-Host ("  Log          : " + $LOG_FILE) -ForegroundColor DarkGray
Write-Host "=================================================" -ForegroundColor Magenta
Write-Host ""
Write-Host "  Commandes :" -ForegroundColor DarkGray
Write-Host "    -Rebuild    : force la recompilation Flutter" -ForegroundColor DarkGray
Write-Host "    -SkipLLM    : ne relance pas LLM/Embeddings" -ForegroundColor DarkGray
Write-Host "    -Debug      : flutter run (hot reload)" -ForegroundColor DarkGray
Write-Host "    -CheckOnly  : verifie et quitte" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Ctrl+C pour arreter la surveillance (services restent actifs)." -ForegroundColor Yellow
Write-Host ""

while ($true) {
    Start-Sleep -Seconds 3
    $servicesDown = @()
    try { $null = Invoke-RestMethod $LLM_HEALTH -TimeoutSec 2 } catch { $servicesDown += "LLM" }
    try { $null = Invoke-RestMethod $EMB_HEALTH -TimeoutSec 2 } catch { $servicesDown += "EMB" }
    try { $null = Invoke-RestMethod $RAG_HEALTH -TimeoutSec 2 } catch { $servicesDown += "RAG" }
    if ($servicesDown.Count -gt 0) {
        $stamp = Get-Date -Format 'HH:mm:ss'
        $joined = $servicesDown -join ', '
        $msg = "[" + $stamp + "] [!] Services DOWN: " + $joined
        Write-Host $msg -ForegroundColor Yellow
    }
}

Stop-Transcript | Out-Null