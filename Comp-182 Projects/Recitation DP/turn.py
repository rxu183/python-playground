def turn(t, m, n, l):
    #I don't know what this function does anymore uh oh.
    
    memo = []
    for i in range(m):
        memo.append([])
        for j in range(n):
            memo.append([])
            for k in range(l):
                memo[i][j].append(0)

    for i in range(m):
        for j in range(n):
            for l in range(l):
                memo[i][j][l] = sum(memo[i][j-1]) + sum(memo[i-1][j])

def main():
    m = 5
    n = 4
    l = 3
    ans =  turn(0, m, n, l) + turn(1, m, n, l)
