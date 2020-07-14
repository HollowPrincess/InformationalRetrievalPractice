
Этот репозиторий создан для выполнения заданий по информационному поиску.

**1. Задание: булева поисковая машина.**

Программа запускается с помощью файла **first_task_search_engine/app.py**.

Постановка задачи в файле: **first_task_search_engine/First_task_formulation.md**

___Краткая формулировка здания___: построить инвертированный индекс на коллекции документов объемом не менее 50 Мб и реализовать булев поиск, ранжирование по числу совпадений слов запроса со словами в документе.

Описание решения в файле: **first_task_search_engine/README.md**

Примеры запросов и ответов: **first_task_search_engine/examples.ipynb**

**2. Задание: ранжирование с помощью векторной модели.**

___Краткая формулировка здания___: улучшить ранжирование за счет введения векторных представлений. Выполняется булев поиск и для наденнных документов по эмбедингам вычисляется косинусная мера близости эмбеддингу запроса. Также добавлено исправление опечаток.
Для выполнения надо запустить скрипт **first_task_search_engine/app.py**.

Первый запуск может занять много времени, так как при этом выполняется построение эмбеддингов документов и загрузка моделей.



___Все задания написаны с аннотированием типов и соблюдением PEP8.___
