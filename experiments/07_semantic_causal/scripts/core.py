import re
import math
import numpy as np
import pandas as pd
from collections import Counter

LOCATION_ORDER = ["Home", "Work", "School", "Leisure", "Transport", "Other"]

# Official MixIT_it_dictionary.xlsx household_monthly_net_income mapping
INCOME_CODE_TO_LABEL = {
    "1": "Less than 1000€",
    "2": "1000€-1499€",
    "3": "1500€-1999€",
    "4": "2000€-2499€",
    "5": "2500€-2999€",
    "6": "3000€-3499€",
    "7": "3500€-3999€",
    "8": "4000€-4999€",
    "9": "More or equal than 5000€",
    "Unknown": "Unknown",
}

PARTICIPANT_COLS = [
    "part_id",
    "part_age_exact",
    "part_gender",
    "hh_size",
    "presence_work",
    "presence_school",
    "educational_attainment",
    "region_nuts1",
    "occupation",
    "household_monthly_net_income",
]

CONTACT_COLS = [
    "cont_id",
    "cnt_age_exact",
    "cnt_gender",
    "frequency_multi",
    "phys_contact",
    "distance",
    "relationship",
    "setting",
    "location_multi",
]

MASTER_SCHEMA_COLS = PARTICIPANT_COLS + CONTACT_COLS


def clean_val(val, is_int=False):
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
    s = clean_val(val)
    if s == "Unknown":
        return "Other"
    s_lower = s.lower()
    for loc in LOCATION_ORDER:
        if loc.lower() == s_lower or loc.lower() in s_lower:
            return loc
    return "Other"


def encode_age_diff(part_age_str, cont_age_str):
    try:
        p_age = int(float(part_age_str))
        c_age = int(float(cont_age_str))
        diff = c_age - p_age
        if diff <= -15:
            return "Younger"
        elif diff >= 15:
            return "Older"
        else:
            return "Peer"
    except (ValueError, TypeError):
        return "Unknown"


def encode_gender(g_str):
    g = str(g_str).strip().upper()
    if g in ["F", "FEMALE"]:
        return "Female"
    if g in ["M", "MALE"]:
        return "Male"
    return "Unknown"


def decode_gender(sem_g):
    s = str(sem_g).strip().upper()
    if s.startswith("F"):
        return "F"
    if s.startswith("M"):
        return "M"
    return "Unknown"


def encode_work_attendance(work_str, occ_str):
    """
    Per MixIT_it_dictionary.xlsx: NA in presence_work identifies respondents
    whose occupation is different from 'Employed'.
    """
    w = clean_val(work_str)
    if w != "Unknown":
        return w
    occ = clean_val(occ_str)
    if occ != "Employed":
        return "Not applicable"
    return "Unknown"


def encode_school_attendance(sch_str, occ_str):
    """
    Per MixIT_it_dictionary.xlsx: NA in presence_school identifies respondents
    whose occupation is different from 'Student'.
    """
    s = clean_val(sch_str)
    if s != "Unknown":
        return s
    occ = clean_val(occ_str)
    if occ != "Student":
        return "Not applicable"
    return "Unknown"


def encode_frequency(freq_val):
    """Maps 1..14 to '1/14 days' .. '14/14 days' per MixIT_it_dictionary.xlsx."""
    c = clean_val(freq_val, is_int=True)
    if c == "Unknown":
        return "Unknown"
    return f"{c}/14 days"


def decode_frequency(sem_freq):
    s = str(sem_freq).strip()
    m = re.match(r"^(\d+)", s)
    if m:
        val = int(m.group(1))
        if 1 <= val <= 14:
            return str(val)
    return "Unknown"


def encode_phys(phys_val):
    s = clean_val(phys_val).lower()
    if s in ["true", "1", "yes"]:
        return "Yes"
    if s in ["false", "0", "no"]:
        return "No"
    return "Unknown"


def decode_phys(sem_phys):
    s = str(sem_phys).strip().lower()
    if s.startswith("yes") or s == "true":
        return "True"
    if s.startswith("no") or s == "false":
        return "False"
    return "Unknown"


def encode_distance(dist_val, phys_yes_no):
    if phys_yes_no == "Yes":
        return "Physical contact"
    d = clean_val(dist_val)
    if d.lower().startswith("pyshical") or d.lower().startswith("physical") or d.lower().startswith("pychical"):
        return "Physical contact"
    return d


def decode_distance(sem_dist, decoded_phys):
    """Restores exact master_data.csv spelling ('Pyshical contact') when Physical Contact is True."""
    if decoded_phys == "True":
        return "Pyshical contact"
    s = clean_val(sem_dist)
    if s.lower().startswith("physical") or s.lower().startswith("pyshical") or s.lower().startswith("pychical"):
        return "Pyshical contact"
    for valid_d in ["Between 1 and 2 meters", "Less than 1 meter", "More than 2 meters"]:
        if valid_d.lower() in s.lower():
            return valid_d
    return "Unknown"


def extract_participant_demos(first_row):
    occ = clean_val(first_row.get("occupation"))
    raw_work = clean_val(first_row.get("presence_work"))
    raw_sch = clean_val(first_row.get("presence_school"))
    inc_code = clean_val(first_row.get("household_monthly_net_income"), is_int=True)

    return {
        "part_id": str(first_row["part_id"]).strip(),
        "age": clean_val(first_row.get("part_age_exact"), is_int=True),
        "gender": clean_val(first_row.get("part_gender")),
        "hh_size": clean_val(first_row.get("hh_size"), is_int=True),
        "work_today": raw_work,
        "work_attendance_sem": encode_work_attendance(raw_work, occ),
        "school_today": raw_sch,
        "school_attendance_sem": encode_school_attendance(raw_sch, occ),
        "occupation": occ,
        "education": clean_val(first_row.get("educational_attainment")),
        "income": inc_code,
        "income_sem": INCOME_CODE_TO_LABEL.get(inc_code, "Unknown"),
        "region": clean_val(first_row.get("region_nuts1")),
    }


def build_prompt(demos):
    return (
        "Generate the daily epidemiological contact diary for the following participant:\n"
        "[PARTICIPANT]\n"
        f"Age: {demos['age']} | Gender: {encode_gender(demos['gender'])} | "
        f"Household Size: {demos['hh_size']} | Occupation: {demos['occupation']} | "
        f"Work Attendance: {demos['work_attendance_sem']} | "
        f"School Attendance: {demos['school_attendance_sem']} | "
        f"Education: {demos['education']} | Household Income: {demos['income_sem']} | "
        f"Region: {demos['region']}"
    )


def format_causal_completion(demos, part_df):
    contact_rows = part_df[
        part_df["cont_id"].notna()
        & (~part_df["cont_id"].astype(str).str.strip().str.lower().isin(["", "nan", "unknown", "none"]))
    ]

    contacts = []
    for _, row in contact_rows.iterrows():
        c_loc = normalize_location(row.get("location_multi"))
        c_rel = clean_val(row.get("relationship"))
        c_set = clean_val(row.get("setting")).lower()
        if c_set not in ["indoor", "outdoor"]:
            c_set = "Unknown"

        c_age = clean_val(row.get("cnt_age_exact"), is_int=True)
        c_age_diff = encode_age_diff(demos["age"], c_age)
        c_gen = encode_gender(row.get("cnt_gender"))
        c_freq = encode_frequency(row.get("frequency_multi"))
        c_phys = encode_phys(row.get("phys_contact"))
        c_dist = encode_distance(row.get("distance"), c_phys)

        try:
            age_sort_key = int(c_age)
        except ValueError:
            age_sort_key = 999

        is_cohab = (c_rel.lower() == "cohabitant")

        contacts.append({
            "loc": c_loc,
            "rel": c_rel,
            "setting": c_set,
            "age_diff": c_age_diff,
            "age": c_age,
            "gender": c_gen,
            "freq": c_freq,
            "phys": c_phys,
            "dist": c_dist,
            "is_cohab": is_cohab,
            "_sort_key": (
                LOCATION_ORDER.index(c_loc) if c_loc in LOCATION_ORDER else 99,
                0 if is_cohab else 1,
                c_rel,
                age_sort_key,
            ),
        })

    contacts.sort(key=lambda x: x["_sort_key"])

    counts = Counter(c["loc"] for c in contacts)
    n_cohab = sum(1 for c in contacts if c["is_cohab"])
    n_non_cohab = len(contacts) - n_cohab

    plan_header = (
        f"[PLAN]\n"
        f"Total: {len(contacts)} | Cohabitants: {n_cohab} | Non-cohabitants: {n_non_cohab} | "
        f"Home: {counts['Home']} | Work: {counts['Work']} | School: {counts['School']} | "
        f"Leisure: {counts['Leisure']} | Transport: {counts['Transport']} | Other: {counts['Other']}"
    )

    if not contacts:
        contacts_block = "[CONTACTS]\n- None"
    else:
        lines = []
        for c in contacts:
            lines.append(
                f"- Location: {c['loc']} | Relationship: {c['rel']} | Setting: {c['setting']} | "
                f"Age Difference: {c['age_diff']} | Age: {c['age']} | Gender: {c['gender']} | "
                f"Frequency: {c['freq']} | Physical Contact: {c['phys']} | Distance: {c['dist']}"
            )
        contacts_block = "[CONTACTS]\n" + "\n".join(lines)

    return f"{plan_header}\n{contacts_block}"


PLAN_REGEX = re.compile(
    r"Total:\s*(\d+)\s*\|\s*Cohabitants:\s*(\d+)\s*\|\s*Non-cohabitants:\s*(\d+)\s*\|\s*"
    r"Home:\s*(\d+)\s*\|\s*Work:\s*(\d+)\s*\|\s*School:\s*(\d+)\s*\|\s*"
    r"Leisure:\s*(\d+)\s*\|\s*Transport:\s*(\d+)\s*\|\s*Other:\s*(\d+)",
    re.IGNORECASE,
)

CONTACT_LINE_REGEX = re.compile(
    r"-\s*Location:\s*(.*?)\s*\|\s*Relationship:\s*(.*?)\s*\|\s*Setting:\s*(.*?)\s*\|\s*"
    r"Age Difference:\s*(.*?)\s*\|\s*Age:\s*(.*?)\s*\|\s*Gender:\s*(.*?)\s*\|\s*"
    r"Frequency:\s*(.*?)\s*\|\s*Physical Contact:\s*(.*?)\s*\|\s*Distance:\s*(.*?)(?:\r?\n|$)"
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
        c_loc = normalize_location(m.group(1))
        c_rel = clean_val(m.group(2))
        c_set = clean_val(m.group(3)).lower()
        if c_set not in ["indoor", "outdoor"]:
            c_set = "Unknown"

        raw_diff = clean_val(m.group(4)).lower()
        if "young" in raw_diff:
            c_gap = "Younger_Gen"
        elif "old" in raw_diff:
            c_gap = "Older_Gen"
        elif "peer" in raw_diff:
            c_gap = "Peer"
        else:
            c_gap = "Unknown"

        c_age = clean_val(m.group(5), is_int=True)
        c_gen = decode_gender(m.group(6))
        c_freq = decode_frequency(m.group(7))
        c_phys = decode_phys(m.group(8))
        c_dist = decode_distance(m.group(9), c_phys)

        rows.append({
            "part_id": str(part_id),
            "Location of Contact": c_loc,
            "Contact Setting": c_set,
            "Relationship to Participant": c_rel,
            "Age_Gap": c_gap,
            "Contact Age": c_age,
            "Contact Gender": c_gen,
            "Contact Frequency": c_freq,
            "Physical Contact": c_phys,
            "Distance during Contact": c_dist,
            "is_zero_contact": False,
            "valid_syntax": True,
            "declared_total": declared_total,
        })
    return rows