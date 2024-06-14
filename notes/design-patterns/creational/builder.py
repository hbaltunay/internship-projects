"""
TR:
Builder deseni, karmaşık nesnelerin adım adım oluşturulmasını sağlayan yaratıcı bir tasarım desenidir. Bu desen,
nesnenin oluşturulma sürecini ve nihai temsilini ayırarak, aynı oluşturma sürecinin farklı temsillerini üretmeyi
mümkün kılar. Builder deseni, genellikle karmaşık nesnelerin oluşturulmasını gerektiren durumlarda kullanılır ve
kodun daha okunabilir ve yönetilebilir olmasını sağlar.

Builder deseni, bir Director sınıfı ve bir veya daha fazla Builder arayüzü içerir. Director sınıfı, oluşturma sürecini
yönetir ve Builder arayüzü, nesne parçalarının oluşturulması için metodlar sağlar. Her ConcreteBuilder sınıfı,
Builder arayüzünü uygular ve karmaşık nesnenin belirli bir temsilini oluşturur.
"""