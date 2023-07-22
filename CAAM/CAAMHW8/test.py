import numpy as np

temp = np.array([[1, 2],
          [5, 4]])


temp1 = np.array([[2, 2],
                 [5, 5]])
ans = np.linalg.svd(temp)
ans1 = np.linalg.svd(temp1)

print(ans)
print(ans1)