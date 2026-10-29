import requests, json
r = requests.post("http://127.0.0.1:8765/execute_tool", json={
    "tool_name": "report_self_repair_rollback",
    "args": {"name": "generate_payroll_slip"},
})
print("CLEANUP:", json.dumps(r.json(), indent=2, ensure_ascii=False))
r = requests.post("http://127.0.0.1:8765/execute_tool", json={
    "tool_name": "payroll_slip",
    "args": {"nom": "Jean Ndong", "salaire_brut": 2500000.0, "mois": "mars 2026", "primes": 200000.0},
})
print("PAYROLL:", json.dumps(r.json(), indent=2, ensure_ascii=False))
