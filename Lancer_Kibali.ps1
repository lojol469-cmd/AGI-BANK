$ErrorActionPreference = 'SilentlyContinue'
$ROOT = "D:\kibali-IA"
$LLAMA = "C:\Users\BELIKANON\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe"
$MODEL_LLM = "$ROOT\models\Qwen3-8B-Q4_K_M.gguf"
$MODEL_EMB = "$ROOT\models\bge-m3-Q4_K_M.gguf"
$PYTHON = "D:\kibali-IA-env\Scripts\python.exe"

Write-Host "🧹 [1/4] Nettoyage des anciens processus..." -ForegroundColor Red
Stop-Process -Name "llama-server" -Force
Stop-Process -Name "streamlit" -Force
Start-Sleep -Seconds 1

Write-Host "🚀 [2/4] Lancement du LLM (Qwen3-8B) sur le port 8080..." -ForegroundColor Cyan
Start-Process -FilePath $LLAMA -ArgumentList @("-m", $MODEL_LLM, "--jinja", "--port", "8080", "-ngl", "99", "-c", "8192") -WindowStyle Minimized

Write-Host "🧠 [3/4] Lancement des Embeddings (BGE-M3) sur le port 8081..." -ForegroundColor Cyan
Start-Process -FilePath $LLAMA -ArgumentList @("-m", $MODEL_EMB, "--embedding", "--port", "8081", "-ngl", "99", "-c", "2048") -WindowStyle Minimized

Write-Host "   [..] Chargement des modèles en VRAM (Patientez ~15s)..." -ForegroundColor Gray
$ready = $false
for ($i=0; $i -lt 30; $i++) {
    Start-Sleep 1
    try {
        $r1 = Invoke-RestMethod "http://127.0.0.1:8080/health" -TimeoutSec 1
        $r2 = Invoke-RestMethod "http://127.0.0.1:8081/health" -TimeoutSec 1
        if ($r1.status -eq "ok" -and $r2.status -eq "ok") { $ready = $true; break }
    } catch {}
}

if (-not $ready) { 
    Write-Host "❌ [ERREUR] Les serveurs llama n'ont pas démarré." -ForegroundColor Red
    Read-Host "Appuyez sur Entrée pour quitter..."
    exit 
}
Write-Host "   [OK] Modèles chargés !" -ForegroundColor Green

Write-Host "🖥️ [4/4] Lancement de Kibali-IA (Streamlit)..." -ForegroundColor Yellow
$env:PYTHONIOENCODING = "utf-8"
& $PYTHON -m streamlit run "$ROOT\app.py" --server.port 8504