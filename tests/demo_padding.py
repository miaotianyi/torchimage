

class temp:
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        return iter(self.data)


a = temp([1,2,3,4])
for x in a:
    print(x)

from torchimage.utils import NdSpec
from itertools import repeat
print("=" * 15)
a = NdSpec(2)

print(a[::-1])

print(list(a[::-1]))
