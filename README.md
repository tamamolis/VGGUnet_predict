## Предиктим семантическую сегментацию
### how to use

- кладём оригинал картинки в папку orig
- в папке orig может быть только одна картинка!
- в терминале пишем cd folder_where_project_exist/
- в терминале пишем python 3 main.py
- сегментированная и склеенная картинка будет в папке crop/
- в папке crop/ может быть только одна картинка!
- в терминале пишем python 3 CRF.py
- картинка с постпроцессом CRF будет в папке result/ 

P.S. сначала надо скомпилить модель с помощью файлика model.py с нужным кол-вом классов, чтобы получить .h5 файл
