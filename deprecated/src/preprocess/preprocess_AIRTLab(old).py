# Seperate AIRTLab into 3 folders ("high-level violence", "low-level violence", "non-violence")

from pathlib import Path
import shutil

try:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
except NameError:
    # Jupyter notebook fallback
    BASE_DIR = Path.cwd().parent.parent

# ====== CONFIG ======
# Dataset root
ROOT = BASE_DIR / "data" / "raw" / "violence-detection-dataset"

# CSV files
CSV_NONVIOLENT = ROOT / "nonviolent-action-classes.csv"
CSV_VIOLENT    = ROOT / "violent-action-classes.csv"

# Output directory
OUT = BASE_DIR / "data" / "processed" / "violence-detection-dataset"

COPY_FILES = True   # True = copy, False = move
# ====================

# Labels from the paper table (make them lowercase)
NON_VIOLENCE_LABELS = {
    "handshake", "highfive", "hug", "jump", "walk", "greet",
    # dataset sometimes uses variants:
    "handgestures", "friendly punch"
}

LOW_LEVEL_LABELS = {
    "push", "slap", "stifle", "fight", "kick", "punch"
}

HIGH_LEVEL_LABELS = {
    "shoot", "stab", "club"
}

# We'll normalize by splitting on ",".
def normalize_actions(action_str: str) -> set[str]:
    return {a for a in action_str.split(",") if a}

def read_csv_map(csv_path: Path) -> dict[str, set[str]]:
    """
    Expected lines like:
    FILE; ACTION CLASSES
    1.mp4;hug,highfive
    2.mp4;highfive,greet
    """
    mapping = {}
    with csv_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # skip header lines
            if line.lower().startswith("file"):
                continue
            # allow ';' delimiter between file and actions
            if ";" in line:
                file_part, action_part = line.split(";", 1)
            else:
                continue
            filename = file_part.strip()
            actions = normalize_actions(action_part)
            mapping[filename] = actions
    return mapping

def category_for_clip(is_violent_folder: bool, actions: set[str]) -> str:
    """
    Decide which of the 3 folders:
      - high-level violence
      - low-level violence
      - non-violence
    """
    if not is_violent_folder:
        # non-violent folder clips go to non-violence
        return "non-violence"

    # violent folder: use actions to split high vs low
    if actions & HIGH_LEVEL_LABELS:
        return "high-level violence"
    if actions & LOW_LEVEL_LABELS:
        return "low-level violence"

    # fallback: if violent folder but no matched label, put into low-level (safer)
    return "low-level violence"

def ensure_out_dirs():
    for cat in ["high-level violence", "low-level violence", "non-violence"]:
        for cam in ["cam1", "cam2"]:
            (OUT / cat / cam).mkdir(parents=True, exist_ok=True)

def copy_or_move(src: Path, dst: Path):
    if COPY_FILES:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)

def main():
    ensure_out_dirs()

    nonviolent_map = read_csv_map(CSV_NONVIOLENT)
    violent_map    = read_csv_map(CSV_VIOLENT)

    # process both top folders
    for top in ["non-violent", "violent"]:
        is_violent = (top == "violent")
        for cam in ["cam1", "cam2"]:
            src_dir = ROOT / top / cam
            if not src_dir.exists():
                print(f"[WARN] Missing folder: {src_dir}")
                continue

            for mp4 in sorted(src_dir.glob("*.mp4")):
                actions = (violent_map if is_violent else nonviolent_map).get(mp4.name, set())
                cat = category_for_clip(is_violent, actions)

                dst = OUT / cat / cam / mp4.name
                copy_or_move(mp4, dst)

    print("Done!")
    print("Output at:", OUT)

if __name__ == "__main__":
    main()
