"""
TR:
Prototype deseni, var olan nesneleri kopyalayarak yeni nesneler oluşturmanın bir yolunu sağlar. Bu desen, özellikle
nesne oluşturma işleminin pahalı veya karmaşık olduğu durumlarda kullanışlıdır. Bir prototip, klonlanabilir ve bu
klonlar, orijinal nesnenin durumunu taşıyarak yeni nesneler olarak kullanılabilir.

Prototype deseni, genellikle bir Prototype arayüzü içerir ve bu arayüz, clone adında bir metod tanımlar. Her somut
prototip sınıfı, bu arayüzü uygular ve kendini kopyalama yeteneğine sahip olur. Klonlama işlemi, genellikle derin kopya
(deep copy) veya yüzeysel kopya (shallow copy) yoluyla gerçekleştirilir.
"""