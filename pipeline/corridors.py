"""V1 corridors (PREDICTION_MODEL_PLAN.md, Step 1).

Placeholder set until the commuter interviews name real routes. Station codes are
NS short codes (see the `stations` table). Each corridor is one direction.
"""

CORRIDORS = [
    ("Rotterdam C -> Amsterdam C", "RTD", "ASD"),
    ("Amsterdam C -> Rotterdam C", "ASD", "RTD"),
    ("Utrecht C -> Amsterdam C", "UT", "ASD"),
    ("Amsterdam C -> Utrecht C", "ASD", "UT"),
    ("Den Haag C -> Utrecht C", "GVC", "UT"),
    ("Utrecht C -> Den Haag C", "UT", "GVC"),
    ("Rotterdam C -> Utrecht C", "RTD", "UT"),
    ("Utrecht C -> Rotterdam C", "UT", "RTD"),
    ("Eindhoven C -> Utrecht C", "EHV", "UT"),
    ("Utrecht C -> Eindhoven C", "UT", "EHV"),
]
