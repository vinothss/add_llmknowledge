"""Build the fixed evaluation sets used from Module 2 onward.

data/eval/lumora_eval.jsonl
    Held-out, human-written questions about Lumora with reference answers.
    Never used for training - it measures how much knowledge a model has.
data/eval/judge_calibration.jsonl
    Candidate answers with HUMAN correctness labels, used to calibrate and test
    the LLM judge. Two sources:
      - "model": real closed/open-book answers from Qwen2.5-0.5B/1.5B-Instruct
        (data/eval/model_answers_labelled.jsonl, labelled by the course author)
      - "handwritten": contrastive examples written by hand (below)
    Split by question: q01-q15 -> "calib" (tune the judge), q16-q30 -> "test".

Run: python scripts/build_eval_sets.py
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "eval"

# (id, category, question, reference answer, source doc title)
EVAL = [
    ("q01", "product", "What is the usable capacity of the Lumora Vault 10?", "10.2 kWh.", "Lumora Vault 10 overview"),
    ("q02", "product", "What battery chemistry does the Lumora Vault use?", "Lithium iron phosphate (LFP).", "Lumora Vault 10 overview"),
    ("q03", "product", "How many Vault 10 units can be connected to one Bridge inverter?", "Up to four units, for a maximum of 40.8 kWh.", "Lumora Vault 10 overview"),
    ("q04", "product", "How much does the Vault 15 weigh?", "141 kg.", "Lumora Vault 15 overview"),
    ("q05", "product", "What is the maximum number of Vault 15 units on a single Bridge inverter?", "Three.", "Lumora Vault 15 overview"),
    ("q06", "product", "How quickly does the Lumora Bridge switch to backup mode during a grid outage?", "In under 20 milliseconds.", "Lumora Bridge hybrid inverter"),
    ("q07", "product", "How many MPPT solar inputs does the Lumora Bridge have?", "Two independent MPPT inputs of up to 600 V each.", "Lumora Bridge hybrid inverter"),
    ("q08", "product", "What does the Storm Watch feature do?", "It automatically switches the system to Backup-Only mode when a severe weather alert is issued for the home's postcode.", "Lumora Home app"),
    ("q09", "product", "What are the three operating modes in the Lumora Home app?", "Self-Powered, Time-Based Control and Backup-Only.", "Lumora Home app"),
    ("q10", "warranty", "How long is the Lumora Vault warranty?", "12 years.", "Vault warranty terms"),
    ("q11", "warranty", "What capacity does Lumora guarantee at the end of the warranty?", "At least 70 percent of the original usable capacity (or after 6,000 full cycles).", "Vault warranty terms"),
    ("q12", "warranty", "What is the warranty period for the Bridge inverter?", "10 years.", "Vault warranty terms"),
    ("q13", "warranty", "How quickly is a warranty claim reviewed?", "A support engineer reviews the diagnostic log within two business days.", "How to make a warranty claim"),
    ("q14", "warranty", "Are customers charged for labour on approved warranty replacements?", "Not during the first five years.", "How to make a warranty claim"),
    ("q15", "support", "What does error code E-101 mean?", "The battery management system detected a cell temperature outside the safe range.", "Error code E-101"),
    ("q16", "support", "What ventilation gap is required around a Vault?", "At least 15 cm.", "Error code E-101"),
    ("q17", "support", "What is the first troubleshooting step for error E-204?", "Power-cycle the Bridge by switching off the AC isolator for 60 seconds.", "Error code E-204"),
    ("q18", "support", "What does error code E-317 indicate?", "The Bridge has lost its internet connection for more than 72 hours.", "Error code E-317"),
    ("q19", "support", "How do you factory-reset the Bridge?", "Hold the service button for 15 seconds until the status light flashes amber three times.", "Resetting the Bridge to factory settings"),
    ("q20", "support", "What is the name of the Wi-Fi network the Bridge broadcasts after a reset?", "LUMORA-SETUP followed by the last four digits of its serial number.", "Resetting the Bridge to factory settings"),
    ("q21", "policy", "Within how many days can unused accessories be returned?", "30 days.", "Customer return policy"),
    ("q22", "policy", "What restocking fee applies to returning an uninstalled Vault?", "10 percent.", "Customer return policy"),
    ("q23", "policy", "How long is Lumora installer certification valid?", "24 months.", "Installer certification programme"),
    ("q24", "policy", "How many days per week may eligible Lumora employees work remotely?", "Up to three days per week.", "Employee remote work policy"),
    ("q25", "policy", "What is the daily meal allowance for Lumora business travel?", "60 euros.", "Employee travel and expenses"),
    ("q26", "company", "Where and when was Lumora Energy founded?", "In Rotterdam in 2017.", "About Lumora Energy"),
    ("q27", "company", "Where is Lumora's main factory?", "Eindhoven.", "About Lumora Energy"),
    ("q28", "engineering", "At what temperature do the Vault's internal fans activate?", "When cell temperature exceeds 38 C.", "Vault thermal design note"),
    ("q29", "engineering", "Why does the Vault heat its cells before charging below 0 C?", "Charging LFP cells below freezing can cause lithium plating.", "Vault thermal design note"),
    ("q30", "faq", "What backup reserve does the Vault keep by default?", "20 percent.", "Does the Vault work during a power cut?"),
]

# (question id, candidate answer, human label)
CALIBRATION = [
    ("q01", "The Vault 10 provides 10.2 kWh of usable energy.", True),
    ("q01", "It has about 13.5 kWh of usable capacity.", False),
    ("q02", "It uses LFP, lithium iron phosphate, cells.", True),
    ("q02", "The Vault uses nickel manganese cobalt (NMC) cells.", False),
    ("q04", "The Vault 15 weighs 141 kilograms.", True),
    ("q04", "Around 98 kg.", False),
    ("q05", "A single Bridge supports up to three Vault 15 units.", True),
    ("q05", "Up to four Vault 15 units per inverter.", False),
    ("q06", "Less than 20 ms.", True),
    ("q06", "It takes about 2 seconds to switch over.", False),
    ("q06", "It switches to backup within seconds of an outage.", False),
    ("q08", "It switches to Backup-Only automatically when a severe weather alert is issued for your area.", True),
    ("q08", "Storm Watch alerts users when severe weather is expected.", False),
    ("q11", "Lumora guarantees a specific capacity level at the end of the warranty.", False),
    ("q11", "At least 70% of the original usable capacity.", True),
    ("q10", "Twelve years.", True),
    ("q10", "The warranty lasts 10 years.", False),
    ("q12", "The Bridge has its own 10-year warranty.", True),
    ("q14", "No labour charge for approved replacements in the first five years.", True),
    ("q14", "Customers always pay for labour.", False),
    ("q15", "E-101 is a cell over- or under-temperature fault detected by the BMS.", True),
    ("q15", "E-101 means the Wi-Fi connection was lost.", False),
    ("q17", "Turn off the AC isolator for a minute to restart the Bridge.", True),
    ("q17", "Replace the battery immediately.", False),
    ("q18", "The Bridge has been offline (no internet) for over 72 hours.", True),
    ("q21", "Within a month - 30 days.", True),
    ("q21", "You have 90 days to return accessories.", False),
    ("q23", "Two years.", True),
    ("q23", "Certification never expires.", False),
    ("q26", "It was founded in Amsterdam in 2015.", False),
    ("q26", "Rotterdam, 2017.", True),
    ("q28", "Above 38 degrees Celsius.", True),
    ("q28", "The fans run all the time.", False),
    ("q30", "I don't have information about Lumora products.", False),
    ("q30", "By default 20% is reserved for outages.", True),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    by_id = {}
    with open(OUT / "lumora_eval.jsonl", "w") as f:
        for qid, cat, q, ref, src in EVAL:
            row = {"id": qid, "category": cat, "question": q, "reference": ref, "source_title": src}
            by_id[qid] = row
            f.write(json.dumps(row) + "\n")
    rows = [{"id": qid, "candidate": cand, "human_label": label, "system": "handwritten", "source": "handwritten"}
            for qid, cand, label in CALIBRATION]
    with open(OUT / "model_answers_labelled.jsonl") as f:
        rows += [dict(json.loads(line), source="model") for line in f]
    with open(OUT / "judge_calibration.jsonl", "w") as f:
        for r in rows:
            q = by_id[r["id"]]
            r.update(question=q["question"], reference=q["reference"],
                     split="calib" if int(r["id"][1:]) <= 15 else "test")
            f.write(json.dumps(r) + "\n")
    n_pos = sum(r["human_label"] for r in rows)
    print(f"{len(EVAL)} eval questions, {len(rows)} calibration items ({n_pos} correct / {len(rows) - n_pos} incorrect)")


if __name__ == "__main__":
    main()
