#Problem Statement


memo = []
def recur(ind1, ind2, s1, s2) -> int:
    #Base Case to Consider:
    if ind1 == -1 or ind2 == -1:
        memo[ind1][ind2] = 0
        return 0
    #DP Storage
    if memo[ind1][ind2] != -1:
        return memo[ind1][ind2]
    #Recurrence:
    if s1[ind1] == s2[ind2]:
        memo[ind1][ind2] = 1 + recur(ind1 - 1, ind2 - 1, s1, s2)
    else:
        memo[ind1][ind2] = max([recur(ind1 - 1, ind2, s1, s2), recur(ind1, ind2 - 1, s1, s2)])
    return memo[ind1][ind2]

def lcs(s1, s2):
    for i in range(len(s1)):
        memo.append([])
        for j in range(len(s2)):
            memo[i].append(-1)
    return recur(len(s1) - 1, len(s2) - 1, s1, s2)


def main():
    subSeq1 = "THISISABOOK982"
    subSeq2 = "THEBOOK3"
    ans = lcs(subSeq1, subSeq2)
    print(ans)
    

main()
