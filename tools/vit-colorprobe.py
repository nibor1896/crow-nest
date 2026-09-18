"""#73 visual regression probe: solid colours, a split image, two bars and a circle.

Every request is ONE image plus one question at temperature 0, so the answer is a
function of the image alone. Before the #73 fix this printed the PREVIOUS image's
answer in every row but the first (red->Black, green->Red, blue->Green, ...), which
reads as a fixed colour permutation and is really a lag of one request.

Run against a live serve:  python3 tools/vit-colorprobe.py   (serve on 127.0.0.1:8099)
Expected: red->Red, green->Green, blue->Blue, white->White, black->Black,
yellow->Yellow, gray->Grey, vertical bar->Vertical, horizontal bar->Horizontal,
white circle->Circle.
"""
import json,zlib,struct,base64,urllib.request
def png(w,h,f):
    raw=b''.join(b'\x00'+b''.join(bytes(f(x,y)) for x in range(w)) for y in range(h))
    def chunk(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
def ask(imgs,q,tag):
    content=[{"type":"image_url","image_url":{"url":'data:image/png;base64,'+base64.b64encode(b).decode()}} for b in imgs]+[{"type":"text","text":q}]
    body={"model":"crow","messages":[{"role":"user","content":content}],"max_tokens":40,"temperature":0,"stream":False}
    r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8099/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=600)
    a=json.loads(r.read())['choices'][0]['message']['content'].strip().replace('\n',' '); print(f'{tag:18s}: {a[:120]}',flush=True)
solid={'red':(230,20,20),'green':(20,200,40),'blue':(20,40,230),'white':(250,250,250),'black':(5,5,5),'yellow':(240,230,30),'gray':(128,128,128)}
for n,c in solid.items(): ask([png(256,256,lambda x,y:c)],"What colour is this image? One word.",n)
ask([png(256,256,lambda x,y:(230,20,20) if x<128 else (20,40,230))],"Left half and right half: name the two colours in order.","left red|right blue")
ask([png(256,256,lambda x,y:(5,5,5) if 100<x<156 else (250,250,250))],"Is the dark bar vertical or horizontal? One word.","vertical bar")
ask([png(256,256,lambda x,y:(5,5,5) if 100<y<156 else (250,250,250))],"Is the dark bar vertical or horizontal? One word.","horizontal bar")
ask([png(256,256,lambda x,y:(250,250,250) if ((x-128)**2+(y-128)**2)<60**2 else (5,5,5))],"What shape is in the middle? One word.","white circle")
