from __future__ import annotations

import re
from typing import Final


# Rules are deliberately table-level rather than value-level.  They are applied to the
# public NHANES file title and collection component before any task is constructed.
# The matched rule is written to the variable catalog so every assignment is auditable.
DOMAIN_RULES: Final[list[tuple[str, str]]] = [
    ("dietary_intake", r"dietary|diet behavior|food security|food frequency|total nutrient|individual foods"),
    ("demographics", r"demographic|income|occupation|housing|acculturation"),
    ("anthropometry", r"body measure|body composition|anthropometr|weight history|dual.energy|dxa"),
    ("respiratory", r"respirat|spirom|asthma|wheez|copd|exhaled nitric|airway"),
    ("cardiovascular", r"blood pressure|cardiovascular|electrocard|pulse|peripheral vascular"),
    ("hematology", r"complete blood count|hematolog|hemoglobin|ferritin|iron status|transferrin|folate.*rbc"),
    ("lipids", r"cholesterol|triglycer|apolipoprotein|lipid"),
    ("glucose_metabolism", r"glucose|glycohemoglobin|insulin|diabetes"),
    ("kidney_urinary", r"albumin.*creatinine|kidney|renal|urine flow|urinary health|urology"),
    ("liver", r"liver|hepatic|fatty liver"),
    ("inflammation_immunology", r"c.reactive|inflamm|immun|allerg|autoantibod|immunoglobulin"),
    ("infectious_disease", r"hepatitis|hiv|cytomegal|herpes|tuberculosis|infection|antibod|enterovirus"),
    ("reproductive_endocrine", r"reproduct|pregnan|fertility|sex steroid|hormone|thyroid|pubert|menstrual"),
    ("nutrition_biomarkers", r"vitamin|folate|iodine|nutritional biochem|fatty acid|carotenoid"),
    ("environmental_exposure", r"arsenic|cadmium|lead|mercur|metal|pestic|herbicide|insecticide|cotinine|nicotine|volatile organic|phthalate|pfas|perfluoro|flame retard|phenol|paraben|environmental|glyphosate|ethylene oxide|consumer product chemical"),
    ("musculoskeletal", r"bone|osteopor|muscle|joint|arthritis|musculoskeletal|physical function"),
    ("oral_health", r"oral|dental|periodontal"),
    ("sensory", r"audiometr|hearing|vision|taste|smell|sensory"),
    ("physical_activity_fitness", r"physical activity|physical fitness|muscle strength|accelerometer"),
    ("mental_health", r"mental health|depress|anxiety|cognitive|neurobehavior|suicid"),
    ("sleep", r"sleep"),
    ("tobacco_alcohol", r"smoking|tobacco|alcohol use|alcohol"),
    ("medication_healthcare", r"prescription medication|drug use|health insurance|hospital utilization|health care|medical conditions|medical history"),
    ("general_biochemistry", r"standard biochemistry|biochemistry profile|albumin|protein|electrolyte"),
]


def classify_table_domain(
    file_id: str,
    component: str,
    collection_component: str = "",
) -> tuple[str, str]:
    """Return a deterministic semantic domain and the rule that assigned it."""

    text = " ".join([str(file_id), str(component), str(collection_component)]).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    for domain, pattern in DOMAIN_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return domain, pattern

    fallback = re.sub(r"[^a-z0-9]+", "_", str(collection_component).lower()).strip("_")
    if not fallback:
        fallback = "unclassified"
    return f"other_{fallback}", "collection_component_fallback"

