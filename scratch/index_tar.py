import urllib.request
import tarfile
import json
import time
from pathlib import Path

url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar"

def get_total_size():
    req = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(req) as resp:
        return int(resp.headers.get('Content-Length', 0))

total_size = get_total_size()
print(f"Total archive size: {total_size / (1024**3):.2f} GB")

index_path = Path("data/Task07_Pancreas/tar_index.json")
members = []
offset = 0

if index_path.exists():
    with open(index_path, "r", encoding="utf-8") as f:
        members = json.load(f)
    if members:
        last = members[-1]
        file_blocks = (last["size"] + 511) // 512
        offset = last["data_offset"] + file_blocks * 512
        print(f"Resuming from member {len(members)}: {last['name']} at offset {offset}")

def fetch_with_retry(start, end, retries=5):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.read()
        except Exception as e:
            if attempt == retries - 1:
                raise e
            time.sleep(1 + attempt)

t0 = time.time()
images_tr_count = sum(1 for m in members if "imagesTr/" in m["name"])
labels_tr_count = sum(1 for m in members if "labelsTr/" in m["name"])

while offset < total_size:
    end = min(offset + 2047, total_size - 1)
    try:
        data = fetch_with_retry(offset, end)
    except Exception as e:
        print(f"Permanent error fetching at {offset}: {e}")
        break

    if not data or len(data) < 512:
        break

    if data[:1024] == b'\x00' * 1024:
        print(f"Reached TAR EOF at offset {offset}")
        break

    try:
        ti = tarfile.TarInfo.frombuf(data[:512], 'utf-8', 'surrogateescape')
    except Exception as e:
        offset += 512
        continue

    file_blocks = (ti.size + 511) // 512
    data_offset = offset + 512

    if ti.name.startswith("Task07_Pancreas/") and not ti.name.startswith("Task07_Pancreas/._"):
        entry = {
            "name": ti.name,
            "size": ti.size,
            "header_offset": offset,
            "data_offset": data_offset,
        }
        members.append(entry)
        
        if "imagesTr/" in ti.name:
            images_tr_count += 1
        elif "labelsTr/" in ti.name:
            labels_tr_count += 1

        if len(members) % 10 == 0 or "labelsTr/" in ti.name or images_tr_count in (1, 10, 50, 100):
            print(f"[{len(members)}] {ti.name} ({ti.size / 1024 / 1024:.2f} MB) | imagesTr: {images_tr_count}, labelsTr: {labels_tr_count}")

        if len(members) % 10 == 0:
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(members, f, indent=2)

    offset = data_offset + file_blocks * 512

# Save final index
with open(index_path, "w", encoding="utf-8") as f:
    json.dump(members, f, indent=2)

print(f"Indexing complete! Total members: {len(members)}, imagesTr: {images_tr_count}, labelsTr: {labels_tr_count}")
