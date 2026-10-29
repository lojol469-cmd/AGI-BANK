import psutil
import matplotlib.pyplot as plt
import time

# Fonction pour obtenir les informations de la CPU
def get_cpu_usage():
    return psutil.cpu_percent(interval=1)

# Fonction pour obtenir les informations de la GPU (approximative)
def get_gpu_usage():
    # Pour Windows, on utilise la bibliothèque nvidia-smi via subprocess
    import subprocess
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu', 'utilization.gpu', '--format', 'csv,nocase,noheader'], stderr=subprocess.STDOUT, text=True)
        return float(result.strip())
    except Exception as e:
        return 0

# Créer un graphique
plt.figure(figsize=(10, 5))
plt.title('Utilisation CPU et GPU')
plt.xlabel('Temps (secondes)')
plt.ylabel('Utilisation (%)')
plt.grid(True)

# Collecter les données
cpu_usage_data = []
gpu_usage_data = []
times = []

for _ in range(10):
    cpu_usage = get_cpu_usage()
    gpu_usage = get_gpu_usage()
    times.append(time.time())
    cpu_usage_data.append(cpu_usage)
    gpu_usage_data.append(gpu_usage)
    time.sleep(1)

# Tracer les données
plt.plot(times, cpu_usage_data, label='CPU')
plt.plot(times, gpu_usage_data, label='GPU')
plt.legend()
plt.show()