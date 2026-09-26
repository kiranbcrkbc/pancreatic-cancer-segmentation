import urllib.request
import tarfile

class BufferedRemoteReader:
    def __init__(self, url, chunk_size=2*1024*1024):
        self.url = url
        self.chunk_size = chunk_size
        self.buffer = b""
        self.buf_start = 0
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            self.total_size = int(resp.headers.get('Content-Length', 0))

    def read_at(self, offset, length):
        # Check if requested range is in buffer
        if not (self.buf_start <= offset and offset + length <= self.buf_start + len(self.buffer)):
            # Fetch new buffer
            self.buf_start = offset
            fetch_len = max(length, self.chunk_size)
            end = min(offset + fetch_len - 1, self.total_size - 1)
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={offset}-{end}'})
            with urllib.request.urlopen(req) as resp:
                self.buffer = resp.read()
        rel_offset = offset - self.buf_start
        return self.buffer[rel_offset:rel_offset + length]

reader = BufferedRemoteReader('https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar')
print(f"Connected to S3. Archive size: {reader.total_size / (1024**3):.2f} GB")

offset = 0
members = []
while offset < 100_000_000 and len(members) < 40:
    block = reader.read_at(offset, 512)
    if not block or len(block) < 512 or block == b'\x00' * 512:
        offset += 512
        continue
    try:
        ti = tarfile.TarInfo.frombuf(block, 'gnu', 'utf-8')
        file_blocks = (ti.size + 511) // 512
        data_offset = offset + 512
        members.append({
            'name': ti.name,
            'size': ti.size,
            'data_offset': data_offset,
            'is_file': ti.isfile(),
            'is_dir': ti.isdir(),
        })
        print(f"[{len(members)}] {ti.name} ({ti.size / 1024 / 1024:.3f} MB) at offset {data_offset}")
        offset = data_offset + file_blocks * 512
    except Exception as e:
        offset += 512

print("Finished scanning first members.")
