from backend.db.query_patient import get_patient_clinical, get_patient_consensus

patient_id = "210259070"

clinical = get_patient_clinical(patient_id)
consensus = get_patient_consensus(patient_id)

print("=== clinical ===")
print(clinical)
print("clinical sex:", None if not clinical else clinical.get("sex"))
print("clinical gender:", None if not clinical else clinical.get("gender"))

print("\n=== consensus ===")
print(consensus)
print("consensus gender:", None if not consensus else consensus.get("gender"))