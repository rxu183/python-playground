memo = []

def main():
    prices = [0, 5, 2, 3, 0]
    print(maxRevenue(prices, 4))

def maxRevenue(prices, n):
    for index in range(n+1):
        memo.append(-1)
    return rodRecur(prices, n)

def rodRecur(prices, n):
    if memo[n] != -1:
        return memo[n]
    if n == 0:
        return 0
    if n == 1:
        return prices[1]
    storage = []
    for i in range(n-1):
        storage.append(rodRecur(prices, i+1) + rodRecur(prices, n - (i + 1)))
    maxTest = max(storage)
    if prices[n] > maxTest:
        return prices[n]
    return maxTest

main()