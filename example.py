# -*- coding: utf-8 -*-
"""
Created on 2021
@author: Алексей Сергеевич Сычев, аспирант группы 911А, ассистент кафедры РТС
"""
import numpy as np
import sympy as sym
#import os
data = []
with open('data_table.txt') as f:
    for line in f:
        data.append([float(x) for x in line.split()])
data = np.array(data)
x,y = list(data.T)
from nonlinear_optimize import *
X,SIGMA,K = sym.symbols('X,SIGMA,K')                    # Задание символьных переменных
F=K*sym.exp(-(X/SIGMA)**2)                              # Задание аналитической функции
F_np,dF_np=numpy_func_and_Jacobian((X,),(K,SIGMA),F)    # Получение численных реализаций функции и её производных
eps=0.01                                                # Задание порогового приращения "эпсилон"
succ,(k,sigma),w_history0,err_history_0=optimizeGaussNewton(F_np,dF_np,(x.flatten(),),(2,3),y,eps=eps,debug=True)
print(succ,k,sigma)
succ,(k,sigma),w_history1,err_history_1=optimizeLevenberg(F_np,dF_np,(x.flatten(),),(2,3),y,eps=eps,debug=True,l=0.3)
print(succ,k,sigma)
succ,(k,sigma),w_history2,err_history_2=optimizeLevenbergMarquardt(F_np,dF_np,(x.flatten(),),(2,3),y,eps=eps,debug=True,l=0.3)
print(succ,k,sigma)