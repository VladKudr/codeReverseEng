import numpy as np
from tracker import board as B
from tracker.field import FieldMarks, FieldProjector
d="data/jobs/1032b8d28d03"
tr=B.load_tracks(d+"/tracks.npz")
with np.load(d+"/inputs.npz") as z: A=z["camera"]
M=np.eye(3); inv=[]
for a in A:
    if not np.isnan(a).any(): M=np.vstack([a,[0,0,1]])@M
    inv.append(np.linalg.inv(M))
L,W=55.0,35.0
base=np.array([[-150,500],[1450,500],[1370,420],[-60,420]],float)
def proj(c):
    P=FieldProjector(FieldMarks(0,c,0),1.0,inv,L,W)
    xs=[];ys=[]
    for i,(ids,boxes) in enumerate(zip(tr.ids,tr.boxes)):
        b=np.asarray(boxes,float).reshape(-1,4)
        if not len(b): continue
        x,y=P(i,(b[:,0]+b[:,2])/2,b[:,3]); xs+=list(x); ys+=list(y)
    return np.array(xs),np.array(ys),P
x0,y0,P=proj(base)
ok=~np.isnan(x0)
print("NaN share %.2f"%(1-ok.mean()))
ins=(x0>-1)&(x0<L+1)&(y0>-1)&(y0<W+1)
print("inside share %.2f"%ins[ok].mean())
# метры на пиксель по вертикали в разных строках кадра 0
for v in (425,440,460,480,500,560,650):
    a=P(0,640,v); b=P(0,640,v+1)
    print(f"row {v}: 1px down = {np.hypot(b[0]-a[0],b[1]-a[1]):.2f} m, pos=({float(a[0]):.1f},{float(a[1]):.1f})")
for v in (440,470):
    a=P(0,640,v); b=P(0,641,v); print(f"row {v}: 1px right = {np.hypot(b[0]-a[0],b[1]-a[1]):.2f} m")
# чувствительность к ошибке пользователя: сдвиг дальних углов на 3 px
for k,dv in [(2,3),(3,3),(0,10)]:
    c=base.copy(); c[k,1]+=dv
    x1,y1,_=proj(c); m=ok&~np.isnan(x1)
    print(f"corner {k+1} +{dv}px: median shift {np.median(np.hypot(x1-x0,y1-y0)[m]):.1f} m, p90 {np.percentile(np.hypot(x1-x0,y1-y0)[m],90):.1f} m")
