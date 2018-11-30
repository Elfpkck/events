#!/usr/bin/env python
# coding: utf-8

# <center>
# <img src="../../img/ods_stickers.jpg">
# ## Открытый курс по машинному обучению
# Авторы материала: программист-исследователь Mail.ru Group, старший преподаватель Факультета Компьютерных Наук ВШЭ Юрий Кашницкий и Data Scientist в Segmento Екатерина Демидова. Материал распространяется на условиях лицензии [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Можно использовать в любых целях (редактировать, поправлять и брать за основу), кроме коммерческих, но с обязательным упоминанием автора материала.

# # <center>Тема 1. Первичный анализ данных с Pandas</center>

# **[Pandas](http://pandas.pydata.org)** — это библиотека Python, предоставляющая широкие возможности для анализа данных. С ее помощью очень удобно загружать, обрабатывать и анализировать табличные данные с помощью SQL-подобных запросов. В связке с библиотеками `Matplotlib` и `Seaborn` появляется возможность удобного визуального анализа табличных данных.

# In[1]:


import numpy as np
import pandas as pd


# Данные, с которыми работают дата саентисты и аналитики, обычно хранятся в виде табличек — например, в форматах `.csv`, `.tsv` или `.xlsx`. Для того, чтобы считать нужные данные из такого файла, отлично подходит библиотека Pandas.
# 
# Основными структурами данных в Pandas являются классы `Series` и `DataFrame`. Первый из них представляет собой одномерный индексированный массив данных некоторого фиксированного типа. Второй - это двухмерная структура данных, представляющая собой таблицу, каждый столбец которой содержит данные одного типа. Можно представлять её как словарь объектов типа `Series`. Структура `DataFrame` отлично подходит для представления реальных данных: строки соответствуют признаковым описаниям отдельных объектов, а столбцы соответствуют признакам.

# ---------
# 
# ## Демонстрация основных методов Pandas 
# 

# ### Series

# **Создание объекта Series из 5 элементов, индексированных буквами:**

# In[3]:


salaries = pd.Series([400, 300, 200, 250], 
              index = ['Andrew', 'Bob', 
                       'Charles', 'Ann']) 
print(salaries)                                                                 


# In[4]:


salaries[salaries > 250]


# **Тип данных и переход к numpy**

# In[7]:


# type(salaries)
salaries.values


# **Индексирование возможно в виде s.Name или s['Name'].**

# In[8]:


print(salaries.Andrew == salaries['Andrew']) 


# In[9]:


salaries['Carl'] = np.nan
salaries


# In[12]:


salaries.fillna(salaries.median(), inplace=True)


# In[13]:


salaries


# ### Создание и изменение

# **Перейдём к рассмотрению объектов типа DataFrame. Такой объект можно создать из массива numpy, указав названия строк и столбцов.**

# In[14]:


df1 = pd.DataFrame(np.random.randn(5, 3), 
                   index=['o1', 'o2', 'o3', 'o4', 'o5'], 
                   columns=['f1', 'f2', 'f3'])
df1


# **Альтернативным способом является создание DataFrame из словаря numpy массивов или списков.**

# In[57]:


df2 = pd.DataFrame({'A': np.random.random(5), 
                    'B': ['a', 'b', 'c', 'd', 'e'], 
                    'C': np.arange(5) > 2})
df2


# **Тип данных и переход к numpy**

# In[19]:


type(df2)
type(df2.values)


# **Обращение к элементам (или целым кускам фрейма):**

# In[20]:


df2.at[3, 'B']


# **Изменение элементов и добавление новых:**

# In[21]:


df2.at[2, 'B'] = 'f'
df2


# #### Обработка пропущенных значений

# In[22]:


df1.at['o2', 'A'] = 2
df1.at['o4', 'C'] = np.nan
df1


# **Булева маска для пропущенных значений (True - там, где был пропуск, иначе - False):**

# In[23]:


pd.isnull(df1)


# **dropna**

# In[27]:


df1.dropna(how='all', axis=1)


# **Пропуски можно заменить каким-то значением.**

# In[28]:


df1.fillna(0)


# ### Чтение из файла и первичный анализ

# Прочитаем данные и посмотрим на первые 5 строк с помощью метода `head`:

# In[29]:


df = pd.read_csv('telecom_churn.csv')


# In[32]:


df.head()


# В Jupyter-ноутбуках датафреймы `Pandas` выводятся в виде вот таких красивых табличек, и `print(df.head())` выглядит хуже.
# 
# Кстати, по умолчанию `Pandas` выводит всего 20 столбцов и 60 строк, поэтому если ваш датафрейм больше, воспользуйтесь функцией `set_option`:

# In[33]:


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)


# А также укажем значение параметра `presicion` равным 2, чтобы отображать два знака после запятой (а не 6, как установлено по умолчанию.

# In[34]:


pd.set_option('precision', 2)


# **Посмотрим на размер данных, названия признаков и их типы**

# In[36]:


df.shape


# Видим, что в таблице 3333 строки и 20 столбцов. Выведем названия столбцов:

# In[41]:


df.columns


# In[44]:


df.index


# Чтобы посмотреть общую информацию по датафрейму и всем признакам, воспользуемся методом **`info`**:

# In[45]:


print(df.info())


# `bool`, `int64`, `float64` и `object` — это типы признаков. Видим, что 1 признак — логический (`bool`), 3 признака имеют тип `object` и 16 признаков — числовые.
# 
# **Изменить тип колонки** можно с помощью метода `astype`. Применим этот метод к признаку `Churn` и переведём его в `int64`:

# In[46]:


df['Churn'] = df['Churn'].astype('int64')


# Метод **`describe`** показывает основные статистические характеристики данных по каждому числовому признаку (типы `int64` и `float64`): число непропущенных значений, среднее, стандартное отклонение, диапазон, медиану, 0.25 и 0.75 квартили.

# In[47]:


df.describe()


# Чтобы посмотреть статистику по нечисловым признакам, нужно явно указать интересующие нас типы в параметре `include`. Можно также задать `include`='all', чтоб вывести статистику по всем имеющимся признакам.

# In[48]:


df.describe(include=['object', 'bool'])


# Для категориальных (тип `object`) и булевых (тип `bool`) признаков  можно воспользоваться методом **`value_counts`**. Посмотрим на распределение нашей целевой переменной — `Churn`:

# In[55]:


df['Total day minutes'].value_counts(dropna=False)


# 2850 пользователей из 3333 — лояльные, значение переменной `Churn` у них — `0`.
# 
# Посмотрим на распределение пользователей по переменной `Area code`. Укажем значение параметра `normalize=True`, чтобы посмотреть не абсолютные частоты, а относительные.

# In[52]:


df['Area code'].value_counts(normalize=False)
df['Area code'].value_counts(normalize=True)


# ### Сортировка
# 
# `DataFrame` можно отсортировать по значению какого-нибудь из признаков. В нашем случае, например, по `Total day charge` (`ascending=False` для сортировки по убыванию):

# In[53]:


df.sort_values(by='Total day charge', 
               ascending=False).head()


# Сортировать можно и по группе столбцов:

# In[61]:


df.sort_values(by=['Churn', 'Total day charge'], 
               ascending=[True, False]).head()


# ### Индексация и извлечение данных

# `DataFrame` можно индексировать по-разному. В связи с этим рассмотрим различные способы индексации и извлечения нужных нам данных из датафрейма на примере простых вопросов.
# 
# Для извлечения отдельного столбца можно использовать конструкцию вида `DataFrame['Name']`. Воспользуемся этим для ответа на вопрос: **какова доля нелояльных пользователей в нашем датафрейме?**

# **Разные способы обращения к столбцам**

# In[69]:


df.Churn.head() # автодополнение
df['Churn'].head() # лучше при использовании констант с именами столбцов и можно получить датафрейм df[['Churn']]
df[['Churn']]


# In[70]:


df['Churn'].mean()


# 14,5% — довольно плохой показатель для компании, с таким процентом оттока можно и разориться.

# Очень удобной является логическая индексация `DataFrame` по одному столбцу. Выглядит она следующим образом: `df[P(df['Name'])]`, где `P` - это некоторое логическое условие, проверяемое для каждого элемента столбца `Name`. Итогом такой индексации является `DataFrame`, состоящий только из строк, удовлетворяющих условию `P` по столбцу `Name`. 
# 
# Воспользуемся этим для ответа на вопрос: **каковы средние значения числовых признаков среди нелояльных пользователей?**

# In[75]:


df


# In[78]:


df[(df['Churn'] == 1) & (df['Total day charge'] > 20)]


# **Важно отличать `and` от `&`, а `or` от `|` и не забывать про скобки в условиях**

# Скомбинировав предыдущие два вида индексации, ответим на вопрос: **сколько в среднем в течение дня разговаривают по телефону нелояльные пользователи**?

# In[79]:


df[df['Churn'] == 1]['Total day minutes'].mean()


# **Какова максимальная длина международных звонков среди лояльных пользователей (`Churn == 0`), не пользующихся услугой международного роуминга (`'International plan' == 'No'`)?**

# In[83]:


df[(df['Churn'] == 0) & (df['International plan'] == 'No')]['Total intl minutes'].max()


# Датафреймы можно индексировать как по названию столбца или строки, так и по порядковому номеру. Для индексации **по названию** используется метод **`loc`**, **по номеру** — **`iloc`**.
# 
# В первом случае мы говорим _«передай нам значения для id строк от 0 до 5 и для столбцов от State до Area code»_, а во втором — _«передай нам значения первых пяти строк в первых трёх столбцах»_. 
# 
# В случае `iloc` срез работает как обычно, однако в случае `loc` учитываются и начало, и конец среза.

# In[86]:


df1


# In[87]:


df1.loc['o3':'o5', 'f2':'A']


# In[88]:


df.loc[0:5, 'State':'Area code']
# df.loc['0':'5', 'State':'Area code'] # то же самое


# In[89]:


df.iloc[0:5, 0:3]


# Метод `ix` индексирует и по названию, и по номеру, но он вызывает путаницу, и поэтому был объявлен устаревшим (deprecated).

# **Строки и столбцы датафрейма могут быть представлены сериями**

# In[94]:


# df.loc[3, :]
# type(df.loc[3, :])
# df.loc[:, 'Area code']
type(df.loc[:, 'Area code'])


# Если нам нужна первая или последняя строчка датафрейма, пользуемся конструкцией `df[:1]` или `df[-1:]`:

# In[95]:


df[-1:]


# ### Применение функций: `apply`, `map` и др.

# **Применение функции к каждому столбцу:**

# In[98]:


df.apply(np.max)


# Метод `apply` можно использовать и для того, чтобы применить функцию к каждой строке. Для этого нужно указать `axis=1`.

# **Применение функции к каждой ячейке столбца**
# 
# Допустим, по какой-то причине нас интересуют все люди из штатов, названия которых начинаются на 'W'. В данному случае это можно сделать по-разному, но наибольшую свободу дает связка `apply`-`lambda` – применение функции ко всем значениям в столбце.

# In[99]:


df[df['State'].apply(lambda state: state[0] == 'W')].head()


# Метод `map` можно использовать и для **замены значений в колонке**, передав ему в качестве аргумента словарь вида `{old_value: new_value}`:

# In[100]:


d = {'No' : False, 'Yes' : True}
df['International plan'] = df['International plan'].map(d)
df.head()


# Аналогичную операцию можно провернуть с помощью метода `replace`:

# In[101]:


df = df.replace({'Voice mail plan': d})
df.head()


# ### Группировка данных
# 
# В общем случае группировка данных в Pandas выглядит следующим образом:
# 
# ```
# df.groupby(by=grouping_columns)[columns_to_show].function()
# ```
# 
# 1. К датафрейму применяется метод **`groupby`**, который разделяет данные по `grouping_columns` – признаку или набору признаков.
# 3. Индексируем по нужным нам столбцам (`columns_to_show`). 
# 2. К полученным группам применяется функция или несколько функций.

# **Группирование данных в зависимости от значения признака `Churn` и вывод статистик по трём столбцам в каждой группе.**

# In[102]:


columns_to_show = ['Total day minutes', 'Total eve minutes', 'Total night minutes']

df.groupby(['Churn'])[columns_to_show].describe(percentiles=[0.5])


# Сделаем то же самое, но немного по-другому, передав в `agg` список функций:

# In[110]:


columns_to_show = ['Total day minutes', 'Total eve minutes', 'Total night minutes']

df.groupby(['Churn'])[columns_to_show].agg([np.mean, np.std, np.min, np.max])


# Объект groupby

# In[116]:


for i, item in df.groupby(['Churn']):
#     print(i)
    item
item


# ### Сводные таблицы

# Допустим, мы хотим посмотреть, как наблюдения в нашей выборке распределены в контексте двух признаков — `Churn` и `Customer service calls`. Для этого мы можем построить **таблицу сопряженности**, воспользовавшись методом **`crosstab`**:

# In[111]:


pd.crosstab(df['Churn'], df['International plan'])


# In[112]:


pd.crosstab(df['Churn'], df['Voice mail plan'], normalize=True)


# Мы видим, что большинство пользователей — лояльные и пользуются дополнительными услугами (международного роуминга / голосовой почты).

# Продвинутые пользователи `Excel` наверняка вспомнят о такой фиче, как **сводные таблицы** (`pivot tables`). В `Pandas` за сводные таблицы отвечает метод **`pivot_table`**, который принимает в качестве параметров:
# 
# * `values` – список переменных, по которым требуется рассчитать нужные статистики,
# * `index` – список переменных, по которым нужно сгруппировать данные,
# * `aggfunc` — то, что нам, собственно, нужно посчитать по группам — сумму, среднее, максимум, минимум или что-то ещё.
# 
# Давайте посмотрим среднее число дневных, вечерних и ночных звонков для разных `Area code`:

# In[113]:


df.pivot_table(['Total day calls', 
                'Total eve calls', 
                'Total night calls'], ['Area code'], 
               aggfunc='mean').head(10)


# ### Преобразование датафреймов
# 
# Как и многие другие вещи, добавлять столбцы в `DataFrame` можно несколькими способами.

# Например, мы хотим посчитать общее количество звонков для всех пользователей. Создадим объект `total_calls` типа `Series` и вставим его в датафрейм:

# In[115]:


total_calls = df['Total day calls'] + df['Total eve calls'] +               df['Total night calls'] + df['Total intl calls']
df.insert(loc=len(df.columns), column='Total calls', value=total_calls) 
# loc - номер столбца, после которого нужно вставить данный Series
# мы указали len(df.columns), чтобы вставить его в самом конце
df.head()


# Добавить столбец из имеющихся можно и проще, не создавая промежуточных `Series`:

# In[120]:


df['Total charge'] = df['Total day charge'] + df['Total eve charge'] +                      df['Total night charge'] + df['Total intl charge']
df.head()


# Чтобы удалить столбцы или строки, воспользуйтесь методом `drop`, передавая в качестве аргумента нужные индексы и требуемое значение параметра `axis` (`1`, если удаляете столбцы, и ничего или `0`, если удаляете строки):

# In[126]:


# избавляемся от созданных только что столбцов
# df = df.drop(['Total charge', 'Total calls'], axis=1) 

# df.drop([1, 2]).head() # а вот так можно удалить строчки
# del df['International plan']
df


# ## Сохранение результата

# In[127]:


df.to_csv('df.csv')


# In[128]:


df.to_html('df.html')


# In[ ]:


get_ipython().system('jupyter nbconvert --to script topic1_habr_pandas.ipynb')


# In[ ]:



