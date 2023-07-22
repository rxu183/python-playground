
def mystery(A, k):
    M = []
    O = []
    for element in A:
        M.append(abs(element))
        O.append(0)
    #print(M)
    C = sorted(M)
    n = len(A)
    for fakeI in range(k):
        i = n - fakeI - 1
        O[ n - i] = C[i]
    return O

def test():
    A = [-5.1, 3.5, 0.1, 0, -6.7, -2.6, 0.3]
    k1 = 3
    k2 = 5
    print(mystery(A, k1))
    print(mystery(A, k2))
test()