import numpy as np

def MM09e2(B_V, age):
    ''' 
    Eqn 2 
    http://adsabs.harvard.edu/abs/2009ApJ...695..679M
    '''
    a = 0.50
    b = 0.15
    P = np.sqrt(age) * (np.sqrt(B_V - a)) - b * (B_V - a)
    return P

def MM09e3(B_V, age):
    ''' Eqn 3 '''
    c = 0.77
    d = 0.40
    f = 0.60
    P = age**0.52 * (c * (B_V - d)**f)
    return P

def MH2008(B_V, age):
    '''
    Equations 12,13,14 from Mamajek & Hillenbrand (2008)
    http://adsabs.harvard.edu/abs/2008ApJ...687.1264M

    Coefficients from Table 10
    
    Parameters
    ----------
    B_V (B-V) color
    age in Myr

    Returns
    -------
    period in color

    '''
    a = 0.407
    b = 0.325
    c = 0.495
    n = 0.566

    f = a * np.power(B_V - c, b)
    g = np.power(age, n)

    P = f * g

    return P


def Angus2015(B_V, age):
    '''
    Compute the rotation period expected for a star of a given color (temp) and age

    NOTE: - input Age is in MYr
          - output Period is in days

    Eqn 15 from Angus+2015
    http://adsabs.harvard.edu/abs/2015MNRAS.450.1787A

    '''
    P = (age ** 0.55) * 0.4 * ((B_V - 0.45) ** 0.31)

    return P

def Gordon2019(B_V):
    return 12.2*B_V**1.8