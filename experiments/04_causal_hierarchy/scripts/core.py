import re
import math
import pandas as pd
from collections import Counter

# The 6 exact MixIT social locations from 'location_multi'
LOCATION_ORDER = ["Home", "Work", "School", "Leisure", "Transport", "Other"]


def clean_val(val, is_int=False):
    """Canonical string sanitizer applied identically to Ground Truth and Generated data."""
    if val is None or pd.isna(val):
        return "Unknown"
    s = str(val).strip()
    if s.lower() in ["", "nan", "none", "<na>", "unknown", "null"]:
        return "Unknown"
    if is_int:
        try:
            f = float(s)
            if math.isnan(f):
                return "Unknown"
            return str(max(0, int(round(f))))
        except (ValueError, TypeError):
            return "Unknown"
    if s.endswith(".0") and s[:-2].lstrip("-").isdigit():
        s = s[:-2]
    return s.replace("|", "/").replace("\n", " ").strip()


def normalize_location(val):
    """Maps raw location_multi values to the 6 canonical MixIT locations."""
    s = clean_val(val)
    if s == "Unknown":
        return "Other"
    s_lower = s.lower()
    for loc in LOCATION_ORDER:
        if loc.lower() == s_lower or loc.lower() in s_lower:
            return loc
    return "Other"


def compute_age_gap(part_age_str, cont_age_str):
    """Derives the generational bridge token between Participant Age and Contact Age."""
    try:
        p_age = int(float(part_age_str))
        c_age = int(float(cont_age_str))
        diff = c_age - p_age
        if diff <= -15:
            return "Younger_Gen"
        elif diff >= 15:
            return "Older_Gen"
        else:
            return "Peer"
    except (ValueError, TypeError):
        return "Unknown"


def extract_participant_demos(first_row):
    return {
        "part_id": str(first_row["part_id"]).strip(),
        "age": clean_val(first_row.get("part_age_exact"), is_int=True),
        "gender": clean_val(first_row.get("part_gender")),
        "hh_size": clean_val(first_row.get("hh_size"), is_int=True),
        "work_today": clean_val(first_row.get("presence_work")),
        "school_today": clean_val(first_row.get("presence_school")),
        "occupation": clean_val(first_row.get("occupation")),
        "education": clean_val(first_row.get("educational_attainment")),
        "income": clean_val(first_row.get("household_monthly_net_income")),
        "region": clean_val(first_row.get("region_nuts1")),
    }


def build_prompt(demos):
    return (
        "Generate the daily epidemiological contact diary for the following participant:\n"
        "[PARTICIPANT]\n"
        f"Age: {demos['age']} | Gender: {demos['gender']} | HH_Size: {demos['hh_size']} | "
        f"Work_Today: {demos['work_today']} | School_Today: {demos['school_today']} | "
        f"Occupation: {demos['occupation']} | Education: {demos['education']} | "
        f"Income: {demos['income']} | Region: {demos['region']}"
    )


def format_causal_completion(demos, part_df):
    """
    Serializes a participant's contacts into [PLAN] + [CONTACTS].
    - [PLAN] decomposes total volume across the 6 MixIT locations (location_multi).
    - [CONTACTS] orders fields causally: Location -> Setting (indoor/outdoor) -> Relation -> Age_Gap -> Age...
    """
    contact_rows = part_df[
        part_df["cont_id"].notna()
        & (~part_df["cont_id"].astype(str).str.strip().str.lower().isin(["", "nan", "unknown", "none"]))
    ]

    contacts = []
    for _, row in contact_rows.iterrows():
        c_loc = normalize_location(row.get("location_multi"))
        c_setting = clean_val(row.get("setting"))  # 'indoor' / 'outdoor' / 'Unknown'
        c_rel = clean_val(row.get("relationship"))
        c_age = clean_val(row.get("cnt_age_exact"), is_int=True)
        c_age_gap = compute_age_gap(demos["age"], c_age)
        c_gender = clean_val(row.get("cnt_gender"))
        c_freq = clean_val(row.get("frequency_multi"))
        c_phys = clean_val(row.get("phys_contact"))
        c_dist = clean_val(row.get("distance"))

        try:
            age_sort_key = int(c_age)
        except ValueError:
            age_sort_key = 999

        contacts.append({
            "loc": c_loc,
            "setting": c_setting,
            "rel": c_rel,
            "age_gap": c_age_gap,
            "age": c_age,
            "gender": c_gender,
            "freq": c_freq,
            "phys": c_phys,
            "dist": c_dist,
            "_sort_key": (
                LOCATION_ORDER.index(c_loc) if c_loc in LOCATION_ORDER else 99,
                c_rel,
                age_sort_key,
            ),
        })

    contacts.sort(key=lambda x: x["_sort_key"])

    counts = Counter(c["loc"] for c in contacts)
    plan_header = (
        f"[PLAN]\n"
        f"Total: {len(contacts)} | Home: {counts['Home']} | Work: {counts['Work']} | "
        f"School: {counts['School']} | Leisure: {counts['Leisure']} | "
        f"Transport: {counts['Transport']} | Other: {counts['Other']}"
    )

    if not contacts:
        contacts_block = "[CONTACTS]\n- None"
    else:
        lines = []
        for c in contacts:
            lines.append(
                f"- Location: {c['loc']} | Setting: {c['setting']} | Relation: {c['rel']} | "
                f"Age_Gap: {c['age_gap']} | Age: {c['age']} | Gender: {c['gender']} | "
                f"Freq: {c['freq']} | Phys: {c['phys']} | Dist: {c['dist']}"
            )
        contacts_block = "[CONTACTS]\n" + "\n".join(lines)

    return f"{plan_header}\n{contacts_block}"


PLAN_REGEX = re.compile(
    r"Total:\s*(\d+)\s*\|\s*Home:\s*(\d+)\s*\|\s*Work:\s*(\d+)\s*\|\s*School:\s*(\d+)\s*\|\s*"
    r"Leisure:\s*(\d+)\s*\|\s*Transport:\s*(\d+)\s*\|\s*Other:\s*(\d+)",
    re.IGNORECASE,
)

CONTACT_LINE_REGEX = re.compile(
    r"-\s*Location:\s*(.*?)\s*\|\s*Setting:\s*(.*?)\s*\|\s*Relation:\s*(.*?)\s*\|\s*"
    r"Age_Gap:\s*(.*?)\s*\|\s*Age:\s*(.*?)\s*\|\s*Gender:\s*(.*?)\s*\|\s*"
    r"Freq:\s*(.*?)\s*\|\s*Phys:\s*(.*?)\s*\|\s*Dist:\s*(.*?)(?:\r?\n|$)"
)


def parse_causal_completion(part_id, completion_text):
    text = str(completion_text or "").strip()
    plan_match = PLAN_REGEX.search(text)
    declared_total = int(plan_match.group(1)) if plan_match else None

    if re.search(r"\[CONTACTS\]\s*-\s*None", text, re.IGNORECASE) or declared_total == 0:
        return [{
            "part_id": str(part_id),
            "Location of Contact": "None",
            "Contact Setting": "None",
            "Relationship to Participant": "None",
            "Age_Gap": "None",
            "Contact Age": "None",
            "Contact Gender": "None",
            "Contact Frequency": "None",
            "Physical Contact": "None",
            "Distance during Contact": "None",
            "is_zero_contact": True,
            "valid_syntax": True,
            "declared_total": 0 if declared_total is None else declared_total,
        }]

    matches = list(CONTACT_LINE_REGEX.finditer(text))
    if not matches:
        return [{
            "part_id": str(part_id),
            "Location of Contact": "Unknown",
            "Contact Setting": "Unknown",
            "Relationship to Participant": "Unknown",
            "Age_Gap": "Unknown",
            "Contact Age": "Unknown",
            "Contact Gender": "Unknown",
            "Contact Frequency": "Unknown",
            "Physical Contact": "Unknown",
            "Distance during Contact": "Unknown",
            "is_zero_contact": False,
            "valid_syntax": False,
            "declared_total": declared_total,
        }]

    rows = []
    for m in matches:
        rows.append({
            "part_id": str(part_id),
            "Location of Contact": normalize_location(m.group(1)),
            "Contact Setting": clean_val(m.group(2)),
            "Relationship to Participant": clean_val(m.group(3)),
            "Age_Gap": clean_val(m.group(4)),
            "Contact Age": clean_val(m.group(5), is_int=True),
            "Contact Gender": clean_val(m.group(6)),
            "Contact Frequency": clean_val(m.group(7)),
            "Physical Contact": clean_val(m.group(8)),
            "Distance during Contact": clean_val(m.group(9)),
            "is_zero_contact": False,
            "valid_syntax": True,
            "declared_total": declared_total,
        })
    return rows