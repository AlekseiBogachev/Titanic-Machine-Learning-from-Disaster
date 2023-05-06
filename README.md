# Titanic-Machine-Learning-from-Disaster
**Статус проекта:** планируются доработки

<img alt="Титаник" src="/sources/Titanic.jpg" height="500">


## Краткое описание
Решение для [Titanic ML competition](https://www.kaggle.com/c/titanic): анализ данных, модель, предсказания, результат на Kaggle.

**Цель** - сделать модель, которая, получив данные о пассажире Титаника, предскажет выжил
пассажир в кораблекрушении или нет.

Решение в [тетрадке](/sources/Titanic-Machine-Learning-from-Disaster.ipynb).


## Приёмы, которые демонстрирует проект
1. Исследовательский анализ данных
2. Тестирование статистических гипотез (гипотеза о равенстве средних - с помощью **scipy.stats.ttest_ind**)
3. Заполнение пропусков
4. Разработка новых признаков
5. Манипуляции с данными средствами **pandas**
6. Использование конвейеров (*Pipeline*) из **sklearn**
7. Оценка информативности признаков
7. Работа с моделями градиентных бустингов: **XGBoost**
8. Оценка моделей
9. Автоматизация оценки производительности моделей и вывода результатов (код собран в функции, устранены повторения кода)
10. Настройка их гиперпараметров с **Optuna**
11. Распараллеливание вычислений с помощью **joblib**
11. Запуск проекта совместно с базой данных с помощью *Docker Compose*
12. Оформление графиков с **seaborn** и **matplotlib**
    - График из исследовательского анализа:
    <img alt="График из исследовательского анализа" src="/img/plots_0.png" height="500">
    
    - Графики из раздела с про разработку признаков:
    <img alt="Матрица корреляции" src="/img/plots_1.png" height="900">
    <img alt="Взаимная информация" src="/img/plots_2.png" height="500">
    
    - Кривые ROC выбранных моделей после настройки гиперпараметров:
    <img alt="Взаимная информация" src="/img/plots_3.png" height="500">

13. Логирование с помощью чат-бота в Телеграм.
<img alt="ROC curve" src="/img/log.jpg" height="500">


## Результат на Kaggle
К сожалению, результат не дотянул до медианного.  
<img alt="Результат на Kaggle" src="/sources/scores_on_kaggle.PNG" height="70">

Во время работы над проектром было перепробовано несколько разных моделей, которые показывали на Kaggle целевую метрику в районе 0.77 - 0.78. Модель выбранная для оформления на GitHub неожиданно показала результат ниже. В дальнейшем она будет улучшена.

Возможные способы улучшения результата:
- удалить выбросы в данных;
- разработать более информативные признаки;
- избавиться от дисбаланса классов с помощью апсемплинга или даунсемплинга, учесть дисбаланс с помощью весов классов;
- продолжить настройку гиперпараметров и побороть оставшийся оверфитинг;
- попробовать использовать более мощный классификатор;
- попробовать объединить классификаторы в ансамбль.


## Запуск проекта
В проекте реализована оптимизация гиперпараметров с **Optuna**, распаралленная
на несколько потоков ("воркеров"), при этом, чтобы сократить количество 
устанавливаемых библиотект, **optuna-distributed** не использовалась. 
Это решение накладывает ограничения на запуск: необходимо сохранять 
промежуточные результаты оптимизации в базе данных.

К **Optuna** для сохранения промежуточных результатов подключается база данных
*PostgreSQL* в строке
```Python
optuna_storage = optuna.storages.RDBStorage('postgresql://postgres:password@postgres/')
```
*Anaconda* и *PostgreSQL* объединены в одно приложение с помощью *Docker Compose*
и предполагается, что запуск проекта будет выполняться именно при помощи него.
Однако, проект можно запустить как в дистрибутиве *Anaconda*, так и в отдельном 
*Docker*-контейнере. Для этого придётся либо самостоятельно подключить к **Optuna**
своё хранилище, способное работать с многопоточной оптимизацией (*SQLite* - не 
очень хорошее решение), либо убрать распараллеливание из проекта и, при 
необходимости, хранилище с промежуточными результатами оптимизации.

### Запуск с *Docker Compose*
Для первого запуска проекта необходимо выполнить следующие шаги:
- запустить терминал из папки с проектом (папка, где лежит **docker-compose.yml**);
- выполнить команду `docker-compose up -d` и дождаться пока *Docker Compose* всё
подготовит;
- выполнить команду `docker-compose exec anaconda /bin/bash`, чтобы попасть в 
командную строку контейнера с *Anaconda*;
- выполнить `conda activate titanic`, чтобы активировать окружение проекта;
- выполнить `jupyter notebook --notebook-dir=/projects --ip='*' --port=8888 --no-browser --allow-root`, чтобы запусить *jupyter notebook* на порте 8888.
- Проект можно останавить, например командой `docker-compose down`(выполнять из папки
с **docker-compose.yml**).

Про запуск *Anaconda* в *Docker* можно почитать [здесь](https://docs.anaconda.com/anaconda/user-guide/tasks/docker/).

### Запуск в *Docker*
Для запуска проекта в *Docker* необходимо выполнить следующие шаги (предполагается, что
командная строка запущена из каталога с проектом):
- собрать образ контейнера с помощью команды `docker build -t titanic . ` (образу будет присвоен тег `titanic`);
- создать и запустить образ с помощью команды `docker run -i -t -p 8888:8888 -v "$(pwd):/projects" --name titanic titanic`
(контейнеру будет присвоено имя `titanic`), после исполнения этой команды порт 8888 хоста будет связан с 
портом 8888 контейнера, текущая директория будет связана ([bind-mount](https://docs.docker.com/storage/bind-mounts/)) 
с каталогом `/projects` контейнера, и будет запущена командная строка контейнера;
- активировать окружение проекта командой `conda activate titanic`;
- запустить *jupyter notebook* на порте 8888 командой
`jupyter notebook --notebook-dir=/projects --ip='*' --port=8888 --no-browser --allow-root`;
- в командной строек появится сылка на сервер *jupyter*, начинающаяся на 127.0.0.1:8888, 
необходимо прейти по ней, чтобы попасть на сервер *jupyter*.

Команда `docker run` каждый раз создаёт новый контейнер, делать это необязательно. Чтобы начать работу с ранее 
созданным контейнером можно использовать команду `docker start -i titanic`, где `titanic` - имя контейнера.
После выполнения `docker start -i titanic` необходимо снова активировать окружение и запускать *jupyter notebook*.

Про запуск *Anaconda* в *Docker* можно почитать [здесь](https://docs.anaconda.com/anaconda/user-guide/tasks/docker/).

### Запуск в *Anaconda*
Для запуска проекта в установленном дистрибутиве *Anaconda* необходимо:
- переименовать файл **requirements.txt** в **titanic.yml**;
- создать окружение из файла **titanic.yml** с помощью команды `conda env create -f titanic.yml` 
(предполагается, что коммандная строка запущена из каталога с **titanic.yml**);
- активировать окружение с помощью команды `conda activate titanic`.  
После этого с проектом можно будет работать, например с помощью *jupyter notebook*;
- в командной строек появится сылка на сервер *jupyter*, начинающаяся на
127.0.0.1:8888, необходимо прейти по ней, чтобы попасть на сервер *jupyter*.


## Логирование 
В проекте сделано автоматическое логирование в телеграм с **notifiers**. Параметры `notifier.notify`, в 
частности, токен и ID чата должны быть размещены в папке с тетрадкой в файле `notifier_params.pkl`.
Естественно, этот файл отсутствует в репозитории, поэтому для настройки логирования и запуска тетрадки
в исходном виде необходимо создать файл `notifier_params.pkl`, содержащий словарь с параметрами для
**notifiers**. Ниже пример словаря:
```Python
notifier_params = {
    'notifier': 'telegram',
    'token': '**********************************************',
    'chat_id': 0000000000
}
```

Также можно просто удалить все следы логирования и библиотеки **notifiers**.

## Структура репозитория
- **datasets** - набор данных и таблица с результатами, взятые на Kaggle
- **img** - иллюстрации для README.md
- **sources**
    - **Titanic-Machine-Learning-from-Disaster.ipynb** - Тетрадка с решением задачи
    - **Titanic.jpg** и **scores_on_kaggle.PNG** - иллюстрации для тетрадки
    - **submission.csv** - предсказания, отправленные на Kaggle
- **docker-compose.yml** - файл для запуска проекта с *Docker Compose*
- **Dockerfile** - файл для создания контейнера с *Anaconda*
- **requirements.txt** - окружение для запуска проекта
- **LICENSE** - MIT License
- **README.md** - файл с описанием проекта
