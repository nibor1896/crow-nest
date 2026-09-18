"""#73 multi-image probe: one, two and three images in a single request.

Checks that the visual rows land in message order - [red, blue] must answer red then
blue and [blue, red] the other way round. Before the fix each image answered for its
predecessor, so the pair came out shifted.

Run against a live serve:  python3 tools/vit-imgprobe.py   (serve on 127.0.0.1:8099)
Expected: red->Red, blue->Blue, checkerboard described as a checkerboard,
[red, blue] -> "Red, Blue", [blue, red] -> blue first.
"""
import json,zlib,struct,base64,urllib.request,sys
def png(w,h,rgb,stripe=None):
    raw=b''
    for y in range(h):
        row=b'\x00'
        for x in range(w):
            c=rgb
            if stripe and (x//stripe[0]+y//stripe[0])%2==0: c=stripe[1]
            row+=bytes(c)
        raw+=row
    def chunk(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')
def url(b): return 'data:image/png;base64,'+base64.b64encode(b).decode()
red=png(224,224,(220,30,30)); blue=png(224,224,(30,60,220)); chk=png(224,224,(255,255,255),stripe=(28,(0,0,0)))
def ask(parts,q,tag):
    content=[{"type":"image_url","image_url":{"url":url(b)}} for b in parts]+[{"type":"text","text":q}]
    body={"model":"crow","messages":[{"role":"user","content":content}],"max_tokens":60,"temperature":0,"stream":False}
    r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8099/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'}),timeout=600)
    j=json.loads(r.read()); a=j['choices'][0]['message']['content'].strip().replace('\n',' ')
    print(f'{tag}: {a[:160]}')
ask([red],"What is the dominant colour of this image? Answer with one word.","red alone")
ask([blue],"What is the dominant colour of this image? Answer with one word.","blue alone")
ask([chk],"Describe this image in five words.","checkerboard alone")
ask([red,blue],"Two images. Name the colour of the first, then the colour of the second.","red then blue")
ask([blue,red],"Two images. Name the colour of the first, then the colour of the second.","blue then red")
ask([chk,red,blue],"Three images. For each, one word: its colour or pattern.","chk red blue")
