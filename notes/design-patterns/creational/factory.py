"""
TR:
Factory deseni, nesne oluşturma işlemini gerçekleştiren bir metodun soyutlanmasını sağlayan yaratıcı bir tasarım
desenidir. Bu desen, nesnelerin oluşturulmasını sınıfların kendilerinden ziyade fabrika metodlarına devreder. Bu sayede,
nesne oluşturma süreci, uygulamanın geri kalan kısmından izole edilir ve nesne türlerinin değiştirilmesi veya
genişletilmesi daha kolay hale gelir.

Factory deseni, genellikle bir Creator sınıfı ve bir veya daha fazla Product arayüzü içerir. Creator sınıfı,
Product arayüzünü uygulayan nesneleri oluşturmak için bir veya daha fazla fabrika metoduna sahiptir. Uygulama,
hangi Product sınıfının örneğinin oluşturulacağını Creator sınıfına bırakır.

"""


from typing import Protocol, Union


class Informative(Protocol):
    def inform(self, information: str) -> str:
        pass


class Weather1:
    def __init__(self) -> None:
        self.informations = {"Temp": 25, "State": "Sunny", "Humidity": 0.6}

    def inform(self, information: str) -> Union[str, int]:
        return self.informations.get(information, "No Info")


class Weather2:
    def __init__(self) -> None:
        self.informations = {"Temp": 15, "State": "Rainy"}

    def inform(self, information: str) -> Union[str, int]:
        return self.informations.get(information, "No Info")


def get_weather(weather: str) -> Informative:
    weathers = {
        "Weather 1": Weather1,
        "Weather 2": Weather2
    }
    return weathers[weather]()


if __name__ == "__main__":

    info_list = ["Temp", "State", "Humidity"]

    w1 = get_weather("Weather 1")
    w2 = get_weather("Weather 1")

    for info in info_list:
        print(f"Weather 1 {info}:", w1.inform(info))
        print(f"Weather 2 {info}:", w2.inform(info))
