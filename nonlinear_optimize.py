# -*- coding: utf-8 -*-
"""
Created on 2021
@author: Алексей Сергеевич Сычев, аспирант группы 911А, ассистент кафедры РТС
"""
import numpy as np
import sympy as sym
import math
def std_err(x,y,f,w):
    return np.sqrt(np.mean((f(x,w)-y)**2))
def numpy_func_and_Jacobian(X,W,F):
    if type(X)!=tuple:
        raise IOError('Первый аргумент функции символьного расчёта матрицы Якобиана должен быть кортеж символьных аргументов функции!')
    if not np.prod(list(map(lambda x:type(x)==sym.core.symbol.Symbol,X))):
        raise IOError('Элементы списка в первом аргументе функции символьного расчёта матрицы Якобиана должны иметь тип sympy.core.symbol.Symbol!')
    if type(W)!=tuple:
        raise IOError('Второй аргумент функции символьного расчёта матрицы Якобиана должен быть кортеж символьных параметров функции!')
    if not np.prod(list(map(lambda x:type(x)==sym.core.symbol.Symbol,W))):
        raise IOError('Элементы списка символьных аргументов должны иметь тип sympy.core.symbol.Symbol!')
    return (sym.lambdify([X,W],F,'numpy'),tuple(map(lambda Wn:sym.lambdify([X,W],sym.diff(F,Wn),'numpy'),W)))
def check_args_for_approx(F_np,dF_np,x,w,y):
    if not callable(F_np):
        raise IOError('Первый аргумент должен быть численной реализацией аппроксимируещей функции.')
    if type(dF_np)!=tuple:
        raise IOError('Второй аргумент должен быть кортежем численных реализаций производных аппроксимируещей функции.')
    if not np.prod(list(map(lambda x:callable(x),dF_np))):
        raise IOError('Второй аргумент должен быть кортежем символьных аргументов типа sympy.core.symbol.Symbol!')
    if type(x)!=tuple:
        raise IOError('Третий аргумент должен быть кортежем одномерных массивов numpy!')
    if type(w)!=tuple:
        raise IOError('Четвёртый аргумент должен быть кортежем численных параметров!')
    if type(y)!=np.ndarray:
        raise IOError('Пятый аргумент должен быть numpy.ndarray!')
def optimizeGaussNewton(F_np,dF_np,x,w,y,eps=0.0001,Niter=100,debug=False):
    check_args_for_approx(F_np,dF_np,x,w,y)
    Nx=x[0].shape[0]
    w_history=[w]
    err=std_err(x, y, F_np, w)
    err_history=[err]
    success=False
    for m in range(0,Niter):
        J=np.array(list(map(lambda func:func(x,w)*np.ones(Nx),dF_np))).T
        if math.isnan(np.prod(J.flatten())):
            raise IOError('NAN внутри матрицы Якобиана ещё до её обращения!')
        JTJ=J.T@J
        if np.linalg.matrix_rank(JTJ)==1:
            break
        f=F_np(x,w)
        try:
            dw=np.linalg.inv(JTJ)@J.T@(y-f)
            w=w+dw
            err=std_err(x, y, F_np, w)
            err_history.append(err)
            w_history.append(w)
        except:
            break
        if np.prod(np.abs(dw/w)<eps):
            success=True
            break
    if debug:
        w_history=np.array(w_history)
        err_history=np.array(err_history)
        return success,w,w_history,err_history
    else:
        return success,w
def optimizeLevenberg(F_np,dF_np,x,w,y,eps=0.0001,Niter=100,debug=False,l=np.float64(0.001)):
    check_args_for_approx(F_np,dF_np,x,w,y)
    Nx=x[0].shape[0]
    Nw=len(w)
    w_history=[w]
    err=std_err(x, y, F_np, w)
    err_history=[err]
    I=np.diag(np.ones(Nw))
    success=False
    for m in range(0,Niter):
        J=np.array(list(map(lambda func:func(x,w)*np.ones(Nx),dF_np))).T
        if math.isnan(np.prod(J.flatten())):
            raise IOError('NAN внутри матрицы Якобиана ещё до её обращения!')
        JTJ=J.T@J
        f=F_np(x,w)
        try:
            dw=np.linalg.inv(JTJ+l*I)@J.T@(y-f)
            wnew=w+dw
            errnew=std_err(x, y, F_np, wnew)
            err_history.append(errnew)
            w_history.append(wnew)
            if errnew>err:
                if l<10**100:
                    l=l*10
                else:
                    break
            else:
                w=wnew
                if l>10**-100:
                    l=l/10
                else:
                    break
                err=errnew
        except:
            break
        if np.prod(np.abs(dw/w)<eps):
            success=True
            break
    if debug:
        w_history=np.array(w_history)
        err_history=np.array(err_history)
        return success,w,w_history,err_history
    else:
        return success,w
def optimizeLevenbergMarquardt(F_np,dF_np,x,w,y,eps=0.0001,Niter=100,debug=False,l=np.float64(0.001)):
    check_args_for_approx(F_np,dF_np,x,w,y)
    Nx=x[0].shape[0]
    w_history=[w]
    err=std_err(x, y, F_np, w)
    err_history=[err]
    success=False
    for m in range(0,Niter):
        J=np.array(list(map(lambda func:func(x,w)*np.ones(Nx),dF_np))).T
        if math.isnan(np.prod(J.flatten())):
            raise IOError('NAN внутри матрицы Якобиана ещё до её обращения!')
        JTJ=J.T@J
        f=F_np(x,w)
        try:
            dw=np.linalg.inv(JTJ+l*np.diag(np.diag(JTJ)))@J.T@(y-f)
            wnew=w+dw
            errnew=std_err(x, y, F_np, wnew)
            err_history.append(errnew)
            w_history.append(wnew)
            if (errnew>err) or math.isnan(np.prod(wnew)):
                if l<10**100:
                    l=l*10
                else:
                    break
            else:
                w=wnew
                if l>10**-100:
                    l=l/10
                else:
                    break
                err=errnew
        except:
            break
        if np.prod(np.abs(dw/w)<eps):
            success=True
            break
    if debug:
        w_history=np.array(w_history)
        err_history=np.array(err_history)
        return success,w,w_history,err_history
    else:
        return success,w
