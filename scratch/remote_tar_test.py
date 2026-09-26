import urllib.request
import tarfile
import io

class RemoteTarReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.pos = 0
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            self.total_size = int(resp.headers.get('Content-Length', 0))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total_size + offset
        return self.pos

    def readinto(self, b):
        size = len(b)
        if self.pos >= self.total_size:
            return 0
        end = min(self.pos + size - 1, self.total_size - 1)
        req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{end}'})
        with urllib.request.urlopen(req) as resp:
            chunk = resp.read()
            n = len(chunk)
            b[:n] = chunk
            self.pos += n
            return n

reader = RemoteTarReader('https://msd-for-monai.s3-us-west-2.amazonaws.com/Task07_Pancreas.tar')
print(f'Total tar size: {reader.total_size / (1024**3):.2f} GB')

# Let's inspect the first members
offset = 0
found = []
while len(found) < 30 and offset < 50_000_000:
    reader.seek(offset)
    header_data = reader.read(512)
    if not header_data or len(header_data) < 512 or header_data == b'\x00' * 512:
        offset += 512
        continue
    try:
        ti = tarfile.TarInfo.frombuf(header_data, 'gnu', 'utf-8')
        found.append((ti.name, ti.size, offset + 512))
        print(f"Found: {ti.name} | Size: {ti.size / 1024 / 1024:.2f} MB | Data offset: {offset + 512}")
        # Next header is aligned to 512 bytes
        file_blocks = (ti.size + 511) // 512
        offset += 512 + file_blocks * 512
    except Exception as e:
        offset += 512
