import struct, sys, collections
T={0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',9:'Q8_1',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',15:'Q8_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',25:'I16',26:'I32',27:'I64',28:'F64',29:'IQ1_M',30:'BF16',34:'TQ1_0',35:'TQ2_0',39:'MXFP4'}
f=open(sys.argv[1],'rb')
def r(fmt): return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))[0]
def s(): n=r('Q'); return f.read(n).decode('utf-8','replace')
def val(t):
    if t==8: return s()
    if t==9:
        at=r('I'); n=r('Q'); v=[val(at) for _ in range(n)]
        return v if n<=8 else f'[{n} items]'
    return r({0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}[t])
assert f.read(4)==b'GGUF'; ver=r('I'); nt=r('Q'); nkv=r('Q')
print('version',ver,'tensors',nt)
for _ in range(nkv):
    k=s(); t=r('I'); v=val(t)
    if 'tokenizer.ggml' in k and isinstance(v,str) and len(v)>200: v=v[:80]
    print(' ',k,'=',v)
c=collections.Counter(); ex=[]
for _ in range(nt):
    name=s(); nd=r('I'); dims=[r('Q') for _ in range(nd)]; tt=r('I'); off=r('Q')
    c[T.get(tt,f'UNKNOWN_{tt}')]+=1
    if len(ex)<25: ex.append((name,dims,T.get(tt,tt)))
print('tensor types:',dict(c))
for e in ex[:40]: print('  ',*e)
