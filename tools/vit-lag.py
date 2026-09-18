"""#73 lag probe: the SAME image sent several times in a row.

The sharpest form of the bug - a repeated image must give a repeated answer, and a
colour change in the middle must be seen on the request that carries it, not on the
next one. Before the fix this printed four reds as "Black" (the image that preceded
them) and the blue in the middle as the previous red.

Run against a live serve:  python3 tools/vit-lag.py   (serve on 127.0.0.1:8099)
Expected: red x4 -> Red, blue -> Blue, yellow x3 -> Yellow.
"""
import json,zlib,struct,base64,urllib.request
def png(c):
    w=h=256; raw=b''.join(b'\x00'+bytes(c)*w for _ in range(h))
    def ch(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+ch(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+ch(b'IDAT',zlib.compress(raw))+ch(b'IEND',b'')
def ask(c,tag):
    body={"model":"crow","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":'data:image/png;base64,'+base64.b64encode(png(c)).decode()}},{"type":"text","text":"What colour is this image? One word."}]}],"max_tokens":16,"temperature":0,"stream":False}
    r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8099/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=300)
    print(f'{tag}: {json.loads(r.read())["choices"][0]["message"]["content"].strip()[:40]!r}',flush=True)
for k in range(4): ask((230,20,20),f'red #{k+1}')
ask((20,40,230),'blue once')
for k in range(3): ask((240,230,30),f'yellow #{k+1}')
