# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Combine this project's own region boxes with the dataset's native nidus boxes and
LIBS-160K's caption templates into a VQA + visual grounding dataset.

Three ingredients, no LLM, no cross-dataset image matching:
  1. bounding_boxes.csv: where each region crop sits in its whole body source, plus the
     region's own normal/abnormal label (this project's deliverable).
  2. wholeBody{ANT,POST}/{ant,post}/{id}.xml: the dataset's own physician-drawn nidus and
     physiological hot spot boxes on the whole body image (context/dataset.md), Pascal VOC
     format, name is Normal or Abnormal. Not previously used by this project. A region's
     hotspots here are found by containment against our own box, not by re-detecting
     anything.
  3. LIBS-160K's 39 caption templates (context/libs160k.md), reused verbatim by exact
     (region, caption type) lookup, not paraphrased.

One JSON record per bounding_boxes.csv row (so per region per patient per view), plus one
"whole_body" record per bs80k-wholebody-bb/bounding_boxes.csv row (context/wholebody_bbox.md),
so "localize the patient's whole body" style questions are grounded in that box instead of one
region's box.
"""
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

RAW = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-imaging-raw")
BB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-bone_region-bb\bounding_boxes.csv")
WB_CSV = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-wholebody-bb\bounding_boxes.csv")
LIBS = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\libs160k-imaging-raw\LIBS-160K-EN")
OUT = Path(r"C:\Users\lulay\Desktop\wbbs-dataset\bs80k-vqa-grounding\grounding_qa.jsonl")

REGION_PHRASES = {
    "right chest": "chest_R", "left chest": "chest_L",
    "right shoulder joint": "shoulder_R", "left shoulder joint": "shoulder_L",
    "right knee joint": "knee_R", "left knee joint": "knee_L",
    "right elbow joint": "elbow_R", "left elbow joint": "elbow_L",
    "right ankle joint": "ankle_R", "left ankle joint": "ankle_L",
    "pelvis": "pelvis", "vertebrae": "vertebrae", "head": "head",
}
PHRASE_BY_CODE = {code: phrase for phrase, code in REGION_PHRASES.items()}

COMPONENT_PREFIX_TO_REGION = {
    "ankleL": "ankle_L", "ankleR": "ankle_R",
    "chestL": "chest_L", "chestR": "chest_R",
    "elbowL": "elbow_L", "elbowR": "elbow_R",
    "kneeL": "knee_L", "kneeR": "knee_R",
    "shoL": "shoulder_L", "shoR": "shoulder_R",
    "head": "head", "pelvis": "pelvis", "vertbra": "vertebrae",
}


def parse_component(component: str) -> tuple[str, str]:
    for view in ("ANT", "POST"):
        if component.endswith(view):
            return COMPONENT_PREFIX_TO_REGION[component[: -len(view)]], view
    raise ValueError(component)


def classify(t: str) -> str:
    tl = t.lower()
    if tl.startswith("this is an image of"):
        return "description"
    if "does not show" in tl:
        return "normal"
    return "abnormal"


def find_region(t: str) -> str:
    tl = t.lower()
    matches = [code for phrase, code in REGION_PHRASES.items() if phrase in tl]
    assert len(matches) == 1, t
    return matches[0]


def load_caption_lookup() -> dict[tuple[str, str], str]:
    records = [json.loads(l) for l in (LIBS / "train" / "train_texts.jsonl").read_text(encoding="utf-8").splitlines()]
    lookup = {}
    for r in records:
        text = r["text"]
        lookup[(find_region(text), classify(text))] = text
    assert len(lookup) == 39
    return lookup


def load_nidus_boxes() -> dict[tuple[int, str], list[tuple[str, int, int, int, int]]]:
    boxes: dict[tuple[int, str], list] = {}
    for view, xml_dir in [("ANT", RAW / "wholeBodyANT" / "ant"), ("POST", RAW / "wholeBodyPOST" / "post")]:
        for xml_path in xml_dir.glob("*.xml"):
            pid = int(xml_path.stem)
            root = ET.parse(xml_path).getroot()
            entries = []
            for obj in root.findall("object"):
                name = obj.find("name").text
                b = obj.find("bndbox")
                entries.append((
                    name,
                    int(b.find("xmin").text), int(b.find("ymin").text),
                    int(b.find("xmax").text), int(b.find("ymax").text),
                ))
            boxes[(pid, view)] = entries
    return boxes


LOW_PRECISION_REGIONS = {"shoulder_L", "shoulder_R"}


def build_region_boxes(df) -> dict[tuple[int, str], dict[str, list[int]]]:
    """(id, view) -> {region_code: [x, y, w, h]}, every region already located for that
    patient/view, used to find which of this project's own region boxes overlap each other."""
    boxes: dict[tuple[int, str], dict[str, list[int]]] = {}
    for _, row in df.iterrows():
        region_code, view = parse_component(row["component"])
        bbox = [int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])]
        boxes.setdefault((int(row["id"]), view), {})[region_code] = bbox
    return boxes


def overlapping_regions(region_code, bbox, sibling_boxes) -> list[str]:
    x, y, w, h = bbox
    overlaps = []
    for other_code, (ox, oy, ow, oh) in sibling_boxes.items():
        if other_code == region_code:
            continue
        if x < ox + ow and ox < x + w and y < oy + oh and oy < y + h:
            overlaps.append(other_code)
    return sorted(overlaps)


def hotspots_in_box(nidus_list, x, y, w, h) -> list[dict]:
    """Abnormal class nidus boxes only. Normal class boxes in the xml are benign
    physiological uptake, not findings, and must not be handed back as the answer to an
    "abnormal tracer uptake" question."""
    found = []
    for name, xmin, ymin, xmax, ymax in nidus_list:
        if name != "Abnormal":
            continue
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        if x <= cx <= x + w and y <= cy <= y + h:
            found.append({"bbox": [xmin, ymin, xmax - xmin, ymax - ymin], "class": name})
    return found


def build_record(row, caption_lookup, nidus_boxes, region_boxes) -> dict:
    region_code, view = parse_component(row["component"])
    phrase = PHRASE_BY_CODE[region_code]
    diagnosis = row["diagnosis"]
    bbox = [int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])]
    hotspots = hotspots_in_box(nidus_boxes.get((int(row["id"]), view), []), *bbox)
    sibling_boxes = region_boxes[(int(row["id"]), view)]
    region_overlap = overlapping_regions(region_code, bbox, sibling_boxes)

    qa = [
        {"question": f"Where is the patient's {phrase} in this bone scan?", "answer_bbox": bbox},
        {"question": f"Does the patient's {phrase} show abnormal tracer uptake?",
         "answer": "Yes" if diagnosis == "abnormal" else "No",
         "answer_text": caption_lookup[(region_code, diagnosis)]},
    ]
    if hotspots:
        qa.append({"question": f"Where is the abnormal tracer uptake in the patient's {phrase}?",
                   "answer_bboxes": [h["bbox"] for h in hotspots]})

    return {
        "image": f"wholeBody{view}/{int(row['id'])}.jpg",
        "region": region_code,
        "view": view,
        "bbox": bbox,
        "diagnosis": diagnosis,
        "caption_description": caption_lookup[(region_code, "description")],
        "caption_diagnosis": caption_lookup[(region_code, diagnosis)],
        "hotspots": hotspots,
        "low_precision_region": region_code in LOW_PRECISION_REGIONS,
        "region_overlap": region_overlap,
        "qa": qa,
    }


def load_wholebody_labels() -> dict[tuple[int, str], str]:
    """(id, view) -> "normal"/"abnormal", same txt convention as every region folder
    (context/dataset.md), wholeBodyANT.txt/wholeBodyPOST.txt label the whole image, not one
    region: abnormal if any nidus box exists anywhere in it."""
    labels = {}
    for view, folder in [("ANT", "wholeBodyANT"), ("POST", "wholeBodyPOST")]:
        txt_path = RAW / folder / f"{folder}.txt"
        for line in txt_path.read_text().splitlines():
            filename, label = line.strip().split("\t")
            pid = int(filename.removesuffix(".jpg"))
            labels[(pid, view)] = "abnormal" if label == "1" else "normal"
    return labels


def build_wholebody_record(row, wb_labels, nidus_boxes) -> dict:
    """"Can you localize the patient's whole body" style questions, grounded in
    bs80k-wholebody-bb/bounding_boxes.csv (the body's own location inside the fixed scanner
    canvas, context/wholebody_bbox.md) rather than one region's location inside the whole body
    image. No caption_description/caption_diagnosis here, LIBS-160K has no whole body caption
    template to borrow, only region ones, inventing one would break the exact reuse this
    project has held to for every other caption field.

    outlier/anomaly_score carry over the IsolationForest flag as-is, same "flag, never drop"
    convention as low_precision_region/region_overlap on region rows. That flag mixes two
    different things, genuinely bad source images and legitimately small real patients
    (context/wholebody_bbox.md's own visual check found both), so dropping flagged rows would
    silently remove real, usable scans along with the bad ones. A consumer that wants only the
    bad-image kind still has to look, this just narrows down where.
    """
    pid, view = int(row["id"]), row["view"]
    bbox = [int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])]
    diagnosis = wb_labels[(pid, view)]
    hotspots = hotspots_in_box(nidus_boxes.get((pid, view), []), *bbox)

    qa = [
        {"question": "Can you localize the patient's whole body in this bone scan?", "answer_bbox": bbox},
        {"question": "Does this whole body bone scan show abnormal tracer uptake anywhere?",
         "answer": "Yes" if diagnosis == "abnormal" else "No"},
    ]
    if hotspots:
        qa.append({"question": "Where is the abnormal tracer uptake in this whole body bone scan?",
                   "answer_bboxes": [h["bbox"] for h in hotspots]})

    return {
        "image": f"wholeBody{view}/{pid}.jpg",
        "region": "whole_body",
        "view": view,
        "bbox": bbox,
        "diagnosis": diagnosis,
        "hotspots": hotspots,
        "outlier": bool(row["outlier"]),
        "anomaly_score": float(row["anomaly_score"]),
        "qa": qa,
    }


def main():
    caption_lookup = load_caption_lookup()
    nidus_boxes = load_nidus_boxes()
    df = pd.read_csv(BB_CSV)
    region_boxes = build_region_boxes(df)
    wb_df = pd.read_csv(WB_CSV)
    wb_labels = load_wholebody_labels()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    n_with_hotspots = 0
    n_overlap = 0
    n_wb_records = 0
    n_wb_with_hotspots = 0
    with OUT.open("w", encoding="utf-8") as f:
        for _, row in df.iterrows():
            record = build_record(row, caption_lookup, nidus_boxes, region_boxes)
            if record["hotspots"]:
                n_with_hotspots += 1
            if record["region_overlap"]:
                n_overlap += 1
            f.write(json.dumps(record) + "\n")

        for _, row in wb_df.iterrows():
            record = build_wholebody_record(row, wb_labels, nidus_boxes)
            n_wb_records += 1
            if record["hotspots"]:
                n_wb_with_hotspots += 1
            f.write(json.dumps(record) + "\n")

    total = len(df) + n_wb_records
    print(f"saved {OUT}, {total} records ({len(df)} region + {n_wb_records} whole_body)")
    print(f"{n_with_hotspots} region records ({n_with_hotspots / len(df):.1%}) have at least one contained hotspot box")
    print(f"{n_overlap} region records ({n_overlap / len(df):.1%}) overlap at least one sibling region box")
    print(f"{n_wb_with_hotspots} whole_body records ({n_wb_with_hotspots / n_wb_records:.1%}) have at least one contained hotspot box")


if __name__ == "__main__":
    main()
