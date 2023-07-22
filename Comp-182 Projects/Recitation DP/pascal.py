#Dynamic Programming is lol
#a)
memo = []
def recur(n, k):
    #Base Case:
    if k == 0:
        return 1
    if n == 0:
        return 0
    #Memo Case:
    if memo[n][k] != -1:
        return memo[n][k]
    #Recursive Case: Select kth Element:
    return recur(n-1, k ) + recur(n-1, k-1)

def choose(n, k):
    for i in range(n+1):
        memo.append([])
        for j in range(k+1):
            memo[i].append(-1)
    ans = recur(n, k)
    return ans

def choose2(n, k):
    memoUp = []
    for i in range(n):
        memoUp.append(0)
    memoUp[0] = 1
    
    for i in range(n):
        memoUp[1] = i
        for j in range(k):
            k = i +j


def main():
    n = 5
    k = 2
    print(choose(10, 5))

main()