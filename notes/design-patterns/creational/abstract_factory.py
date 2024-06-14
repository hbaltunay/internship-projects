"""
TR:
Abstract Factory, soyut sınıflar kullanarak birbirleriyle ilgili nesnelerin ailesini oluşturmak için bir arayüz sağlayan
yaratıcı bir tasarım desenidir. Bu desen, nesnelerin oluşturulma şeklinden ziyade nesnelerin türlerine odaklanır ve
nesne oluşturma sorumluluğunu uygulamanın geri kalanından soyutlar. Bu, kodunuzun bağımlılıklarını azaltır ve
değişikliklere karşı daha esnek olmasını sağlar.

Abstract Factory deseni, bir dizi soyut sınıf veya arayüz tanımlar ve bu arayüzleri uygulayan somut sınıflar oluşturur.
Her bir soyut sınıf, belirli bir nesne ailesi için bir fabrika görevi görür ve her fabrika, o aileye ait nesneleri
oluşturmak için özelleştirilmiş yöntemler sunar.
"""

from typing import Type


class Media:
    def __init__(self, name:str) -> None:
        self.name = name

    def play(self) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError


class Video(Media):
    def play(self) -> None:
        print("Video is playing.")

    def __str__(self) -> str:
        return f"Video Name: {self.name}"


class Music(Media):
    def play(self) -> None:
        print("Music is playing.")

    def __str__(self) -> str:
        return f"{self.name}"


class MediaPlayer:
    def __init__(self, media_type: Type[Media]) -> None:
        self.media_type = media_type

    def download_media(self, name: str) -> Media:
        media = self.media_type(name)
        print(f"{media} downloaded.")
        return media


if __name__ == "__main__":
    media_player = MediaPlayer(Music)
    media = media_player.download_media("Despacito")
    media.play()
