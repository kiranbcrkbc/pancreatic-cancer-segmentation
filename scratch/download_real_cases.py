import urllib.request
import json
import time
from pathlib import Path
import nibabel as nib
import numpy as np

url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"

def download_file_range(data_offset, size, dest_path, retries=5):
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size == size:
        # Already downloaded
        return True

    end = data_offset + size - 1
    req = urllib.request.Request(url, headers={'Range': f'bytes={data_offset}-{end}'})
    for attempt in range(retries):
        try:
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()
            if len(data) != size:
                raise ValueError(f"Expected {size} bytes, got {len(data)}")
            with open(dest_path, "wb") as f:
                f.write(data)
            dt = time.time() - t0
            print(f"Downloaded {dest_path.name} ({size/1024/1024:.2f} MB) in {dt:.1f}s ({size/1024/1024/dt:.2f} MB/s)")
            return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"Failed downloading {dest_path.name}: {e}")
                return False
            time.sleep(2 + attempt)

def download_patients(patient_ids, index_file="data/Task07_Pancreas/tar_index.json"):
    with open(index_file, "r", encoding="utf-8") as f:
        members = json.load(f)

    name_to_entry = {m["name"]: m for m in members}
    success = {}

    for pid in patient_ids:
        img_name = f"Task07_Pancreas/imagesTr/{pid}.nii.gz"
        lbl_name = f"Task07_Pancreas/labelsTr/{pid}.nii.gz"

        if img_name not in name_to_entry:
            print(f"Warning: {img_name} not yet in tar index")
            continue
        if lbl_name not in name_to_entry:
            print(f"Warning: {lbl_name} not yet in tar index")
            continue

        img_dest = Path("data/Task07_Pancreas/imagesTr") / f"{pid}.nii.gz"
        lbl_dest = Path("data/Task07_Pancreas/labelsTr") / f"{pid}.nii.gz"

        img_ok = download_file_range(name_to_entry[img_name]["data_offset"], name_to_entry[img_name]["size"], img_dest)
        lbl_ok = download_file_range(name_to_entry[lbl_name]["data_offset"], name_to_entry[lbl_name]["size"], lbl_dest)

        if img_ok and lbl_ok:
            try:
                img_nii = nib.load(str(img_dest))
                lbl_nii = nib.load(str(lbl_dest))
                img_data = img_nii.get_fdata()
                lbl_data = lbl_nii.get_fdata()
                labels_present = np.unique(np.round(lbl_data).astype(int))
                print(f"Verified {pid}: Image shape {img_data.shape}, Spacing {img_nii.header.get_zooms()[:3]}, Labels present: {labels_present}")
                success[pid] = {
                    "shape": list(img_data.shape),
                    "spacing": [float(x) for x in img_nii.header.get_zooms()[:3]],
                    "labels": [int(x) for x in labels_present]
                }
            except Exception as e:
                print(f"Verification failed for {pid}: {e}")

    return success

if __name__ == "__main__":
    import sys
    pids = sys.argv[1:] if len(sys.argv) > 1 else ["pancreas_001", "pancreas_004", "pancreas_005"]
    download_patients(pids)
