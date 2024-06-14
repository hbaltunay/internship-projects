"""
TR:
Borg (veya Monostate) deseni, Singleton desenine benzer bir amaç taşır: uygulama boyunca yalnızca bir nesne örneğinin
olmasını sağlamak. Ancak, Singleton’ın aksine, Borg deseni aynı durumu paylaşan birden fazla nesne örneği oluşturur.
Yani, her Borg örneği, aynı durumu paylaşır ve bu nedenle, herhangi bir örnekte yapılan değişiklikler tüm örnekleri
etkiler.

Borg deseninde, tüm örneklerin durumunu (state) saklamak için sınıf seviyesinde bir sözlük kullanılır. Bu sözlük,
__dict__ özelliği üzerinden tüm örnekler tarafından paylaşılır. Yeni bir örnek oluşturulduğunda, bu örneğin __dict__
özelliği, sınıfın paylaşılan durumunu gösterir.
"""


class System:
    _shared_features = {}

    def __init__(self) -> None:
        self.__dict__ = self._shared_features


class Component(System):
    def __init__(self, activity: bool = None) -> None:
        super().__init__()

        if activity:
            self.activity = activity
        else:
            self.activity = False

    def __str__(self) -> str:
        if self.activity:
            return "The system has been started."
        else:
            return "The system has been stopped."


if __name__ == "__main__":
    comp1 = Component()
    comp2 = Component()

    comp1.activity = True
    comp2.activity = False

    print(comp1)
    print(comp2)