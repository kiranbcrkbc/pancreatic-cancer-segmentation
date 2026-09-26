import urllib.request
import tarfile

class FastTarScanner:
    def __init__(self, url):
        self.url = url
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            self.total_size = int(resp.headers.get('Content-Length', 0))

    def fetch_range(self, start, end):
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(req) as resp:
            return resp.read()

scanner = FastTarScanner('https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar')
print(f"Total archive size: {scanner.total_size / (1024**3):.2f} GB")

# Read a 512KB chunk at start
chunk = scanner.fetch_range(0, 524287)
offset = 0
found = []
while offset < len(chunk) - 512:
    block = chunk[offset:offset+512]
    if block == b'\x00' * 512:
        offset += 512
        continue
    try:
        ti = tarfile.TarInfo.frombuf(block, 'utf-8', 'surrogateescape')
        found.append((ti.name, ti.size, offset + 512))
        print(f"Member: {ti.name} | size: {ti.size / 1024:.1f} KB | offset: {offset+512}")
        file_blocks = (ti.size + 511) // 512
        offset += 512 + file_blocks * 512
    except Exception as e:
        offset += 512

print(f"Found {len(found)} members in first 512KB")
