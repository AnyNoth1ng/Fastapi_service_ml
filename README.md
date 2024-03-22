﻿**Предсказание стоимости акции на FastApi**

**Функционал:**

1) /history/{ticker} – вывод цены закрытия за вчера
1) /graph/{ticker} – вывод графика стоимости акции
1) /predict/{ticker} – предсказание цены на завтра
1) /status- проверка работы сервиса

**Что есть в проекте:**

1) Docker
1) Docker\_compose
1) Кэширование Redis (сделано в predict)

Как работает сервис:

Вам надо знать тикер акции и вписать вместо {ticker}. Их можно найти в сервисе Yahoo Finance.

Если вам надо проект запустить локально, то запуск можно сделать с помощью docker compose up
