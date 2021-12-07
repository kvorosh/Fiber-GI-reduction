Комплекс программ для генерации изображений методами сжатых измерений и редукции.
Основные модули:

- compressive_sensing.py — реализованные варианты метода сжатых измерений при
регуляризующем функционале — норме L<sup>1</sup> изображения в базисе
дискретного косинусного преобразования и в базисе Хаара, квадратичной кривизне и двух вариантах
анизотропной полной вариации. Входные данные — сформированная
модель измерения, последовательность показаний собирающего датчика и параметр
регуляризации. Выходные данные — полученная оценка изображения и значения
нормы невязки и регуляризующего функционала.
- fiber_propagation.py — моделирование изменения шаблона освещения,
сформированного SLM или DMD, в результате распространения по оптическому волокну
с заданными свойствами к объекту, с помощью [pyMMF](https://github.com/vongostev/pyMMF).
Входные данные — изображение на входе оптического волокна,
выходные данные — изображение на
выходе оптического волокна.
- measurement_model.py — построение модели измерения, включая
расчет псевдо- или квазислучайных шаблонов освещения и их загрузку из файлов,
моделирование получения данных измерений, а также формирование ФИ традиционным методом.
Входные данные — условия формирования изображений
(например, максимальное обрабатываемое число измерений и размер пикселя), тип
или источник шаблонов освещения. Выходные данные — сформированная модель
измерения.
- processing.py — основная программа, отвечающая за численное моделирование восстановления ФИ,
сравнение результатов различных методов восстановления изображения
(визуально и по квадратичной погрешности) и подбор параметров методов.
- reduction.py — реализованные варианты метода редукции измерений:
линейная редукция, соответствующая отсутствию дополнительной информации об
объекте, и редукция при дополнительной информации о разреженности изображения объекта в выбранном базисе.
В качестве такого базиса могут быть использованы
собственный базис модели интерпретации измерения, базис дискретного косинусного преобразования и базис Хаара.
Входные данные — сформированная модель измерения, последовательность показаний собирающего датчика
и, при редукции измерений при дополнительной информации о разреженности изображения объекта,
параметры алгоритма, включая выбранный базис. Выходные данные — полученная
оценка изображения и, при редукции измерений при дополнительной информации
о разреженности изображения объекта, отношения компонент оценки в выбранном
базисе к их стандартным отклонениям.

Исследование выполнено за счет гранта Российского научного фонда (проект № 21-12-00155).