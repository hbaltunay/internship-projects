"""
TR:

Delegation Pattern, bir nesnenin, başka bir nesneye işi devretmesine olanak tanır. Yani, bir nesne kendi üzerine düşen
bir görevi yerine getirmek yerine, bu görevi başka bir nesneye devreder. Bu, kod tekrarını azaltmaya ve daha temiz,
okunabilir ve bakımı kolay kod yazmaya yardımcı olur.

Bu yaklaşımın avantajları şunlardır:

- Esneklik: Yeni bir dosya türü eklemek istediğinizde, sadece yeni bir uploader sınıfı eklemeniz ve FileUploader
sınıfındaki sözlüğe bu sınıfı eklemeniz yeterli olacaktır.
- Bakım Kolaylığı: Her dosya türünün yükleme işlevselliği ayrı sınıflarda tanımlandığı için, kod üzerinde yapılacak
değişiklikler daha kolay ve izole edilebilir.
- Tek Sorumluluk Prensibi: Her sınıf, sadece kendi dosya türünü yüklemekle sorumlu olduğu için, tek sorumluluk
prensibine uygun bir yapı oluşturulmuş olur.
- Kodun Okunabilirliği: Kod daha modüler ve okunabilir hale gelir, çünkü her sınıfın sorumluluğu ve işlevselliği açıkça
tanımlanmıştır.
"""


class PdfUploader:
    @staticmethod
    def upload(file):
        print(f"<{file}> Uploaded as PDF.")


class DocUploader:
    @staticmethod
    def upload(file):
        print(f"<{file}> Uploaded as Word.")


class XlsUploader:
    @staticmethod
    def upload(file):
        print(f"<{file}> Uploaded as Excel.")


class FileUploader:
    def __init__(self):
        self._uploaders = {
            "pdf": PdfUploader(),
            "doc": DocUploader(),
            "xls": XlsUploader()
        }

    def upload_file(self, file_type, file):
        uploader = self._uploaders.get(file_type)
        if uploader:
            uploader.upload(file)
        else:
            print("Unsupported file type.")


if __name__ == "__main__":
    file_uploader = FileUploader()
    file_uploader.upload_file("pdf", "test.pdf")
