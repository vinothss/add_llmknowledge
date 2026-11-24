"""Build the deliberately messy raw corpus used across the workshop.

Lumora Energy is a FICTIONAL company. Because no base model has ever seen
these facts, learners can measure whether CPT / SFT really injected knowledge.

The raw corpus contains clean documents plus planted problems:
  - exact duplicates, near duplicates
  - HTML-wrapped scrapes with navigation boilerplate
  - PII (names, emails, phone numbers) in support tickets
  - junk/short pages, spammy repetition, non-English pages

Run:  python scripts/build_raw_corpus.py
Writes: data/raw/lumora_corpus.jsonl and data/raw/answer_key.json
"""
from __future__ import annotations

import json
import random
from pathlib import Path

random.seed(7)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw"

# --------------------------------------------------------------------------
# 1. Clean "gold" documents: the knowledge we want the model to learn.
# --------------------------------------------------------------------------
CLEAN = [
    ("product", "Lumora Vault 10 overview",
     "The Lumora Vault 10 is a wall-mounted home battery with 10.2 kWh of usable capacity. "
     "It uses lithium iron phosphate (LFP) cells and delivers 5 kW continuous power with a 7.5 kW peak for 10 seconds. "
     "The Vault 10 weighs 98 kg and is rated IP55, so it can be installed indoors or in a sheltered outdoor location. "
     "Operating temperature range is -10 C to 50 C. Up to four Vault units can be stacked on one Lumora Bridge inverter "
     "for a maximum of 40.8 kWh. The Vault 10 ships with firmware version 4.2 and receives over-the-air updates through the Lumora Home app."),
    ("product", "Lumora Vault 15 overview",
     "The Lumora Vault 15 offers 15.4 kWh of usable capacity and 7 kW continuous power, with a 10 kW peak for 10 seconds. "
     "It is designed for larger homes and small businesses. The Vault 15 weighs 141 kg and must be floor-mounted on the included steel plinth. "
     "Like the Vault 10 it uses LFP chemistry and is rated IP55. A maximum of three Vault 15 units can be connected to a single Bridge inverter. "
     "Vault 10 and Vault 15 units cannot be mixed on the same inverter."),
    ("product", "Lumora Bridge hybrid inverter",
     "The Lumora Bridge is a 10 kW hybrid inverter that connects solar panels, Vault batteries and the grid. "
     "It supports two independent MPPT solar inputs of up to 600 V each. In a grid outage the Bridge switches to backup mode in under 20 milliseconds, "
     "which is fast enough that most appliances do not notice the interruption. The Bridge reports energy data every 5 seconds to the Lumora Home app "
     "over Wi-Fi or Ethernet. A cellular modem is available as the optional Bridge LTE accessory."),
    ("product", "Lumora Home app",
     "Lumora Home is the companion app for iOS and Android. It shows live solar production, battery state of charge and household consumption. "
     "Owners can choose between three operating modes: Self-Powered, Time-Based Control and Backup-Only. "
     "Self-Powered mode maximises use of solar energy. Time-Based Control charges the battery when electricity is cheap and discharges during peak tariffs. "
     "Backup-Only keeps the battery at 100 percent for outages. The Storm Watch feature automatically switches to Backup-Only when a severe weather alert is issued for the home's postcode."),
    ("warranty", "Vault warranty terms",
     "Every Lumora Vault is covered by a 12-year limited warranty. Lumora guarantees that the battery will retain at least 70 percent of its original usable capacity "
     "at the end of the warranty period or after 6,000 full cycles, whichever comes first. The warranty requires installation by a Lumora Certified Installer "
     "and a continuous internet connection for at least 90 percent of the warranty period so that firmware updates and health monitoring can run. "
     "The Bridge inverter carries a separate 10-year warranty."),
    ("warranty", "How to make a warranty claim",
     "To make a warranty claim, open the Lumora Home app, go to Settings, then Support, then Warranty Claim. The app uploads a 30-day diagnostic log automatically. "
     "A support engineer reviews the log within two business days. If a replacement is approved, a Lumora Certified Installer will contact the customer to book a visit. "
     "Customers are never charged for labour on approved warranty replacements during the first five years."),
    ("support", "Error code E-101",
     "Error code E-101 means the battery management system detected a cell temperature outside the safe range. "
     "The Vault will stop charging and discharging until the temperature returns to between 0 C and 45 C. "
     "Check that the ventilation gap of at least 15 cm around the unit is clear. If E-101 appears more than three times in 24 hours, contact Lumora Support."),
    ("support", "Error code E-204",
     "Error code E-204 indicates a communication fault between the Vault and the Bridge inverter. "
     "First, power-cycle the Bridge by switching off the AC isolator for 60 seconds. Then check the CAN cable between the units for damage. "
     "If the error persists after a restart, the installer should run the Bridge Link diagnostic from the installer menu."),
    ("support", "Error code E-317",
     "Error code E-317 means the Bridge lost its internet connection for more than 72 hours. Energy production and backup continue to work normally, "
     "but firmware updates and remote monitoring are paused. Reconnect the Bridge to Wi-Fi from the Lumora Home app under Settings, Network. "
     "Extended loss of connectivity can affect warranty eligibility."),
    ("support", "Resetting the Bridge to factory settings",
     "A factory reset erases all network settings and operating-mode preferences but keeps installer configuration such as grid code and battery count. "
     "To reset, hold the Bridge service button for 15 seconds until the status light flashes amber three times. "
     "After the reset, the Bridge broadcasts a setup Wi-Fi network named LUMORA-SETUP followed by the last four digits of its serial number."),
    ("policy", "Customer return policy",
     "Customers may return accessories such as the Bridge LTE modem or wall brackets within 30 days of delivery for a full refund, provided they are unused and in original packaging. "
     "Vault batteries and Bridge inverters cannot be returned once installed, because installation is a regulated electrical activity. "
     "Uninstalled Vault units may be returned within 14 days, subject to a 10 percent restocking fee to cover hazardous-goods shipping."),
    ("policy", "Installer certification programme",
     "The Lumora Certified Installer programme requires a two-day training course at a Lumora Academy site and a practical assessment. "
     "Certification is valid for 24 months. Installers must complete at least five supervised installations before they can work independently. "
     "Certified installers receive access to the installer menu in the Lumora Home app and priority technical support on a dedicated phone line."),
    ("policy", "Employee remote work policy",
     "Lumora employees in office-eligible roles may work remotely up to three days per week. Core collaboration hours are 10:00 to 15:00 local time. "
     "Employees must use the company VPN and a Lumora-managed laptop for all remote work. Field technicians and Academy trainers are not eligible for remote work "
     "because their roles require on-site presence."),
    ("policy", "Employee travel and expenses",
     "Lumora reimburses business travel booked through the internal travel portal. Economy class is standard for flights under six hours; "
     "premium economy is allowed for longer flights. The daily meal allowance is 60 euros. Expense reports must be submitted within 30 days of the trip, "
     "with receipts for any item over 25 euros. Alcohol is not reimbursable."),
    ("company", "About Lumora Energy",
     "Lumora Energy was founded in 2017 in Rotterdam by battery engineers Ines Valkenburg and Tomas Rheede. "
     "The company designs home energy storage systems and sells them through a network of certified installers in nine European countries. "
     "Lumora's mission is to make every home its own power station. The company operates its main factory in Eindhoven and employs around 1,400 people."),
    ("company", "Lumora sustainability report summary",
     "In its most recent sustainability report Lumora stated that 96 percent of battery materials are recoverable through its take-back programme. "
     "End-of-life Vault units are collected free of charge and sent to a recycling partner in Belgium. "
     "The Eindhoven factory runs on 100 percent renewable electricity, and Lumora targets net-zero operational emissions by 2030."),
    ("engineering", "Vault thermal design note",
     "The Vault enclosure uses passive cooling with an aluminium rear heat sink. Internal fans activate only when cell temperature exceeds 38 C. "
     "Below 0 C, a 120 W internal heater warms the cells before charging is allowed, because charging LFP cells below freezing can cause lithium plating. "
     "Discharging is permitted down to -10 C."),
    ("engineering", "Grid code configuration",
     "During commissioning the installer selects the national grid code in the installer menu. The grid code sets voltage and frequency trip limits "
     "and the reactive power behaviour of the Bridge. Selecting the wrong grid code is a common cause of nuisance tripping. "
     "The grid code can only be changed by a certified installer and every change is logged in the Bridge audit trail."),
    ("faq", "Can I add a Vault later?",
     "Yes. A Vault 10 can be added to an existing Vault 10 system at any time, up to four units per Bridge. "
     "Lumora recommends adding new units within three years of the original installation, because cells of very different ages balance less efficiently. "
     "Adding a unit requires a certified installer and takes about two hours."),
    ("faq", "Does the Vault work during a power cut?",
     "Yes, when paired with a Lumora Bridge the system provides backup power during a grid outage. Backup circuits are chosen by the installer and usually include "
     "the fridge, lighting, internet router and heating controls. By default the Vault keeps a 20 percent reserve for outages, which owners can change in the Lumora Home app."),
]

# --------------------------------------------------------------------------
# 2. Noise generators.
# --------------------------------------------------------------------------
NAV = ("Home | Products | Support | Find an Installer | Cookie settings\n"
       "Skip to main content\n")
FOOTER = ("\n(c) Lumora Energy B.V. All rights reserved. Privacy policy | Terms of use | "
          "Subscribe to our newsletter for the latest offers!")


def html_wrap(title: str, text: str) -> str:
    paras = "".join(f"<p>{s.strip()}.</p>" for s in text.split(". ") if s.strip())
    return (f"<html><head><title>{title}</title><script>trackPage()</script></head><body>"
            f"<nav>{NAV}</nav><div class='content'><h1>{title}</h1>{paras}</div>"
            f"<footer>{FOOTER}</footer></body></html>")


def near_dup(text: str) -> str:
    # Small edits: extra whitespace, an appended boilerplate sentence, one word changed.
    t = text.replace(". ", ".  ", 2)
    t = t.replace(" is ", " is currently ", 1).replace(" are ", " are currently ", 1)
    return t + "\nLast updated by the documentation team."


TICKETS = [
    ("Support ticket 88213",
     "Customer Maria Jansen (maria.jansen@example.nl, +31 6 1234 5678) reported error code E-204 after a thunderstorm. "
     "The agent advised power-cycling the Bridge for 60 seconds, which cleared the fault. Customer confirmed the system is back online."),
    ("Support ticket 88227",
     "Caller David Okafor phoned from 020 7946 0958 about the Vault showing E-101 on a hot afternoon. "
     "The unit was installed in a loft with no ventilation gap. Agent booked installer visit; follow-up email sent to d.okafor@example.co.uk."),
    ("Support ticket 88301",
     "Ms. Sofia Romano asked whether she can add a second Vault 10 four years after installation. "
     "Agent explained that it is possible but cell balancing is less efficient after three years. Contact: sofia.romano@example.it."),
    ("Support ticket 88345",
     "Peter Lindqvist (peter.l@example.se) reported E-317 after changing his home router. "
     "Agent walked him through reconnecting the Bridge in the Lumora Home app under Settings, Network. Callback number +46 70 123 45 67."),
    ("Support ticket 88390",
     "Installer Anna Kowalski, phone +48 601 234 567, asked how to change a grid code after commissioning. "
     "Agent confirmed only certified installers can change it and the change is recorded in the audit trail."),
    ("Support ticket 88412",
     "Customer Jean Dupont, jean.dupont@example.fr, wanted to return an installed Vault 15. "
     "Agent explained installed batteries cannot be returned but offered a free system health check."),
]

JUNK = [
    ("404", "404 Page not found. Click here to go home."),
    ("login", "Please log in to continue."),
    ("cookie", "We use cookies. Accept all cookies?"),
    ("blank", "   "),
]

SPAM = [
    ("promo", ("BEST BATTERY DEALS!!! BUY NOW BUY NOW BUY NOW. " * 12).strip()),
    ("keywords", " ".join(["solar battery cheap solar battery best home battery discount"] * 15)),
]

NON_ENGLISH = [
    ("de", "Die Lumora Vault 10 ist ein wandmontierter Heimspeicher mit 10,2 kWh nutzbarer Kapazitaet. "
           "Sie verwendet Lithium-Eisenphosphat-Zellen und liefert 5 kW Dauerleistung. Die Garantie betraegt zwoelf Jahre."),
    ("es", "La Lumora Vault 15 ofrece 15,4 kWh de capacidad util y 7 kW de potencia continua. "
           "Esta disenada para hogares grandes y pequenas empresas y debe instalarse en el suelo."),
    ("nl", "Lumora Energy is in 2017 opgericht in Rotterdam. Het bedrijf ontwerpt thuisbatterijen en verkoopt "
           "deze via gecertificeerde installateurs in negen Europese landen."),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, key = [], {"clean": [], "exact_dup": [], "near_dup": [], "html": [],
                     "pii": [], "junk": [], "spam": [], "non_english": []}

    def add(kind: str, source: str, title: str, text: str) -> None:
        doc_id = f"doc-{len(rows):03d}"
        rows.append({"id": doc_id, "source": source, "title": title, "text": text})
        key[kind].append(doc_id)

    for cat, title, text in CLEAN:
        add("clean", f"docs/{cat}", title, text)
    for cat, title, text in random.sample(CLEAN, 5):
        add("exact_dup", f"mirror/{cat}", title, text)
    for cat, title, text in random.sample(CLEAN, 5):
        add("near_dup", f"wiki/{cat}", title, near_dup(text))
    for cat, title, text in random.sample(CLEAN, 5):
        add("html", f"web/{cat}", title, html_wrap(title, text))
    for title, text in TICKETS:
        add("pii", "crm/tickets", title, text)
    for title, text in JUNK:
        add("junk", "web/misc", title, text)
    for title, text in SPAM:
        add("spam", "web/ads", title, text)
    for lang, text in NON_ENGLISH:
        add("non_english", f"web/{lang}", f"Localized page ({lang})", text)

    random.shuffle(rows)
    with open(OUT / "lumora_corpus.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (OUT / "answer_key.json").write_text(json.dumps(key, indent=2))
    print(f"Wrote {len(rows)} documents to {OUT/'lumora_corpus.jsonl'}")
    for k, v in key.items():
        print(f"  {k:12s} {len(v)}")


if __name__ == "__main__":
    main()
