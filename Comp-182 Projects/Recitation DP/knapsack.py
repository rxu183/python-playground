import numpy as np
def D2gen(n):

    test = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    test.append(np.array([[i - n/2 , j - n/2],
                                   [k - n/2, l - n/2]]))
    
    for element in test:
        if (np.matmul(element, np.transpose(element)) == np.identity(2)).all():
            print(element)


def kpGreedy (objects) :
    for element in objects:
        return 0
    return 0
  
def main():
    D2gen(20)

main()


