#!/usr/bin/env python
# coding: utf-8

# ПУНКТ 1

# In[51]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[52]:


#2.1
loc, scale, N = 3, 2, 10000
s = np.random.logistic(loc, scale, N)


# ----
# ----
# ----
# ----
# ----

# ПУНКТ 2

# In[53]:


min_s = min(s)
max_s = max(s)
mean_s = s.mean()
uncorrect_variance = np.var(s, ddof = 0) #неисправленная дисперсия
correct_variance = np.var(s, ddof = 1)   #исправленная дисперсия
standart_deviation = np.std(s, ddof = 1)  # или просто извлечь корень из correct_variance (среднеквадр отклонение)


# ![ms.png](attachment:ms.png)

# In[54]:


print('Минимальное значение выборки: ', min_s)
print('Максимальное значение выборки: ', max_s)
print('Выборочное среденее: ', mean_s)
print('Выборочная (неисправленная) дисперсия: ',  uncorrect_variance)
print('Выборочная (исправленная) дисперсия: ',  correct_variance)
print('Среднеквадратическое от исправленной дисперсии: ', standart_deviation)


# -------------------------------------------------------------------------------------------------------------------
# 
# Письменный расчет мат ожидания и дисперсии
# ![image.png](attachment:image.png)

# In[55]:


#пункт 2.3
def analysis(n):
    l, s, n = 3, 2, n
    array = np.random.logistic(l, s, n)
    mean_s = array.mean()                         # выборочное среднее
    correct_variance = np.var(array, ddof = 1)    #исправленная дисперсия
    return [array, mean_s, correct_variance]


# In[56]:


def create_frame(n):
    frame = [analysis(n) for i in range(1000)]                        # сгенерировали 1000 выборок по n элементов
#    print(len(frame[0][0]))
    table_dict = {"array": [frame[i][0] for i in range(1000)],
                 "mean": [frame[i][1] for i in range(1000)],
                 "variance": [frame[i][2] for i in range(1000)]}
    df = pd.DataFrame(table_dict)
    return df


# In[57]:


# построение графиков
def draw_graph(data_frame):
    theoretical_mean = 3  # Математическое ожидание
    theoretical_variance = (np.pi**2 * 2**2) / 3  # Дисперсия 

    # Построение графиков
    plt.figure(figsize=(20, 10))     # задает размер графика (ширина, длина)

    # График выборочного среднего
    plt.plot(data_frame['mean'], label='Выборочное среднее')
    plt.axhline(theoretical_mean, color='red', linestyle='-', label='Теоретическое среднее', linewidth = 5)
    plt.xlabel('Номер реализации', fontsize = 15)
    plt.ylabel('Выборочное среднее', fontsize = 15)
    plt.legend()



    # График исправленной выборочной дисперсии
    plt.figure(figsize=(20, 10))     # задает размер графика (ширина, длина)
    plt.plot(data_frame['variance'], label='Исправленная выборочная дисперсия')
    plt.axhline(theoretical_variance, color='red', linestyle='-', label='Теоретическая дисперсия', linewidth = 5)
    plt.xlabel('Номер реализации', fontsize = 20)
    plt.ylabel('Исправленная выборочная дисперсия', fontsize = 20)
    plt.legend()

    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ПОСТРОЕНИЕ ДЛЯ N = 10

# In[58]:


df_10 = create_frame(10)
print(df_10)
draw_graph(df_10)


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# 
# ПОСТРОЕНИЕ ДЛЯ N = 50

# In[59]:


df_50 = create_frame(50)
print(df_50)


# In[60]:


draw_graph(df_50)


# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------
# 
# ПОСТРОЕНИЕ ДЛЯ N = 100

# In[61]:


df_100 = create_frame(100)
print(df_100)


# In[62]:


draw_graph(df_100)


# -------
# -------
# 
# Для N = 10000

# In[63]:


df_10000 = create_frame(10000)
print(df_10000)


# In[64]:


draw_graph(df_10000)


# ВЫВОД: при увеличении объема выборки(при увеличении N) мат ожидание и дисперсия стремятся к вычисленным вручную значениям (колебания от вычисленной величины начинают уменьшаться) (смотреть внимательно на ось Oy)
# 

# -------
# ------
# 
# 

# In[65]:


#Пункт 2.4

df = create_frame(10000)
average_from_mean = df['mean'].mean()
average_from_variance = df['variance'].mean()

print("Среднее от выборочного среднего: ", average_from_mean)
print("Среднее оn исправленной выборочной дисперсии: ", average_from_variance)


# ВЫВОД: Значения близки друг к другу (и близки к значениям вычисленным вручную)

# -----
# -----
# -----
# -----
# -----

# ПУНКТ 3
# (работа с выборкой из пункта 1)

# In[66]:


from scipy.stats import logistic

#r = 1 + int(np.log2(10000))                   
r = 50

h = (max_s - min_s)/r
interval_boundaries = np.linspace(min_s, max_s, r+1)

#в interval_boundaries сгенерили просто r+1 число
#потом по ним в np.histogram разбили на r+1 интервалов
hist_counts, _ = np.histogram(s, bins = interval_boundaries)  
# his_count - массив, в котором хранится кол-во вхождений в элементов интеравал 
# _ - это массив границ

relative_frequencies = hist_counts / (h * 10000)

plt.figure(figsize=(17, 5))
plt.bar(interval_boundaries[:-1], relative_frequencies, width=h, alpha=0.5, label='Гистограмма')
plt.xlabel('Границы интервалов', fontsize = 20)
plt.ylabel('Относительные частоты', fontsize = 20)
plt.legend()

s_theoretical = np.linspace(min_s, max_s, 10000)
pdf_logistic = logistic.pdf(s_theoretical, loc, scale) # посчитали значения плотности вероятности для каждого значения из выборки
plt.plot(s_theoretical, pdf_logistic, 'r-', label='Логистическое распределение')
#s_theoretical - значения по 0x
#pdf_logistic - значения по 0y

plt.show()


# -----
# -----
# -----

# ПУНКТ 4

# In[67]:


grouped_mean = np.sum((interval_boundaries[:-1] + interval_boundaries[1:]) * 0.5 * hist_counts) / len(s_theoretical)
print(len(interval_boundaries[:-1] + interval_boundaries[1:]))


# PYTHON ARRAYS
# a = [1, 2, 3] 
# b = [4, 5, 6]
# a + b = [1, 2, 3, 4, 5, 6]
# 2 * a = [1, 2, 3, 1, 2, 3]


#NUMPY ARRAYS
# A = [1, 2, 3]
# B = [4, 5, 6]

# A+B:  [5 7 9]
# 2*A:  [2 4 6]

# Вычисление выборочной дисперсии по группированным данным
grouped_variance = np.sum(((interval_boundaries[:-1] + interval_boundaries[1:]) * 0.5 - grouped_mean)**2 * hist_counts) / (len(s_theoretical) - 1)
print("Выборочное среднее для группированных данных: ", grouped_mean)
print("Выборочная дисперсия для группированных данных: ",grouped_variance)
print("----------------------------")
print("Выборочные характеристки из пункта 2")
print('Выборочное среденее: ', mean_s)
print('Выборочная (исправленная) дисперсия: ',  correct_variance)


# ![image-2.png](attachment:image-2.png)
# ![image.png](attachment:image.png)
# 

# ----
# ----
# ----

# ПУНКТ 5

# In[70]:


plt.figure(figsize=(10, 6))
#Эмпирическая функция для груп/негруп данных
import seaborn as sns
sns.kdeplot(s, cumulative=True)
sns.kdeplot(s_theoretical, cumulative=True)



# Параметры распределения
loc, scale = 3, 2

# Создание массива значений для x
x_theoretical = np.linspace(loc - 4 * scale, loc + 4 * scale, 1000)

# Рассчет теоретической функции распределения
cdf_theoretical = logistic.cdf(x_theoretical, loc=loc, scale=scale)

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(x_theoretical, cdf_theoretical, label='Теоретическая функция распределения', color='green')


# ---
# ---
# ---
# ТЕСТИРОВАНИЕ

# In[72]:


a = [1, 2, 3]
b = [4, 5, 6]
print("python")
print("a+b: ", a + b)
print("2*a: ", 2 * a)
print("-----------")
print("numpy")
A = np.array([1, 2, 3])
B = np.array([4, 5, 6])
print("A+B: ", A + B)
print("2*A: ", 2 * A)


# In[ ]:




