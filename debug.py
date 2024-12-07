from multiprocessing import Pool

a = [21,2,2,2,4,2,2,2,2,]

b = [2,2,2,2,1,2,2,4,2]

print(len(a))
print(len(b))

def inner(a):
    res = []
    for a_i in a:
        inter = 0
        for b_i in b:
            inter += a_i*b_i
        res.append(inter)
    return res

def inner_single(a_i):
    inter = 0
    for b_i in b:
        inter += a_i * b_i
    return inter

print(inner(a))


def inner_multiprocess(a):
    with Pool(4) as p:
        return p.map(inner_single, a)

print(inner_multiprocess(a))
