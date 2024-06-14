"""
TR:
Pool tasarım deseni, nesnelerin önceden oluşturulmuş ve hazır bir havuzdan (pool) alınıp kullanılmasını sağlayan bir
yaratıcı tasarım desenidir. Bu desen, özellikle nesne oluşturma maliyeti yüksek olduğunda ve nesneler sık sık
oluşturulup yok edildiğinde kullanışlıdır. Pool deseni, nesneleri yeniden kullanarak performansı artırır ve gereksiz
nesne oluşturma işlemlerini azaltır.

Pool deseni, nesnelerin bir havuzda yönetildiği bir Pool sınıfı içerir. Bu havuz, kullanılmayan nesneleri saklar ve bir
nesneye ihtiyaç duyulduğunda, havuzdan bir nesne alınır. Nesne kullanımı bittikten sonra, nesne havuza geri verilir ve
yeniden kullanılmak üzere saklanır.
"""


class AddData:
    def __init__(self, dataset):
        self._dataset = dataset
        self.data = None

    def __enter__(self):
        if self.data is None and self._dataset:
            self.data = self._dataset.pop()
        return self.data

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.data is not None:
            self._dataset.add(self.data)
            self.data = None

    def __del__(self):
        if self.data is not None:
            self._dataset.add(self.data)
            self.data = None


if __name__ == "__main__":
    dataset = set()
    dataset.add("Test")

    with AddData(dataset) as ds:
        idata = ds
    odata = ds

    print(idata == odata)  # True
