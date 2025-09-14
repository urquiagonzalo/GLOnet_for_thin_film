import torch

# complex number c : (tuple) (c_R, c_I)
# complex 2x2 matrix: C : (tuple) (C11, C12, C21, C22)
# each element in the tuple can be arbitrary tensor

def complex_add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def complex_sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def complex_mul(a, b):
    c_R = a[0] * b[0] - a[1] * b[1]
    c_I = a[0] * b[1] + a[1] * b[0]
    return (c_R, c_I)


def complex_div(a, b):
    return complex_mul(a, complex_inv(b))


def complex_opp(a):
    return (-a[0], -a[1])


def complex_inv(a):
    denominator = a[0] * a[0] + a[1] * a[1]
    a_inv_R = a[0] / denominator
    a_inv_I = -a[1] / denominator
    return (a_inv_R, a_inv_I)


def complex_abs(a):
    return torch.sqrt(a[0] * a[0] + a[1] * a[1])
    

def matrix_mul(A, B):
    C11 = complex_add(complex_mul(A[0], B[0]), complex_mul(A[1], B[2]))
    C12 = complex_add(complex_mul(A[0], B[1]), complex_mul(A[1], B[3]))
    C21 = complex_add(complex_mul(A[2], B[0]), complex_mul(A[3], B[2]))
    C22 = complex_add(complex_mul(A[2], B[1]), complex_mul(A[3], B[3]))
    
    return (C11, C12, C21, C22)


def matrix_inv(A):
    det_A_inv = complex_inv(complex_sub(complex_mul(A[0], A[3]), complex_mul(A[1], A[2])))

    A11_inv = complex_mul(det_A_inv, A[3])
    A12_inv = complex_mul(det_A_inv, complex_opp(A[1]))
    A21_inv = complex_mul(det_A_inv, complex_opp(A[2]))
    A22_inv = complex_mul(det_A_inv, A[0])
    
    return (A11_inv, A12_inv, A21_inv, A22_inv) 

def transfer_matrix_layer(thickness, refractive_index, k, ky, pol):
    '''
    args:
        thickness (tensor): batch size x 1 x 1 x 1
        refractive_index (tensor): batch size x (number of frequencies or 1) x 1 x 1
        k (tensor): 1 x number of frequencies x 1 x 1
        ky (tensor): 1 x number of frequencies x number of angles x 1
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    kx = torch.sqrt(torch.pow(k * refractive_index, 2)  - torch.pow(ky, 2))

    TEpol = -torch.pow(refractive_index, 2)
    TMpol = torch.ones_like(TEpol)

    if pol == 'TM':
        pol_multiplier = TMpol
    elif pol == 'TE':
        pol_multiplier = TEpol
    else:
        pol_multiplier = torch.cat([TMpol, TEpol], dim = -1)
    
    T11_R = torch.cos(kx * thickness)
    T11_I = torch.zeros_like(T11_R)
    
    T12_R = torch.zeros_like(T11_R)
    T12_I = torch.sin(kx * thickness) * k / kx * pol_multiplier
    
    T21_R = torch.zeros_like(T11_R)
    T21_I = torch.sin(kx * thickness) * kx / k / pol_multiplier
    
    T22_R = torch.cos(kx * thickness)
    T22_I = torch.zeros_like(T11_R)
    
    return ((T11_R, T11_I), (T12_R, T12_I), (T21_R, T21_I), (T22_R, T22_I))


def transfer_matrix_stack(thicknesses, refractive_indices, k, ky, pol = 'TM'):
    '''
    args:
        thickness (tensor): batch size x number of layers
        refractive_indices (tensor): batch size x number of layers x (number of frequencies or 1)
        k (tensor): 1 x number of frequencies x 1
        ky (tensor): 1 x number of frequencies x number of angles 
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    N = thicknesses.size(-1)
    numfreq = refractive_indices.size(-1)

    T_stack = ((1., 0.), (0., 0.), (0., 0.), (1., 0.))
    for i in range(N):
        thickness = thicknesses[:, i].view(-1, 1, 1, 1)
        refractive_index = refractive_indices[:, i, :].view(-1, numfreq, 1, 1)
        T_layer = transfer_matrix_layer(thickness, refractive_index, k, ky, pol)
        T_stack = matrix_mul(T_stack, T_layer)
        
    return T_stack


def amp2field(refractive_index, k, ky, pol = 'TM'):
    '''
    args:
        refractive_index (tensor): 1 x (number of frequencies or 1) x 1 x 1
        k (tensor): 1 x number of frequencies x 1 x 1
        ky (tensor): 1 x number of frequencies x number of angles x 1
        pol (str): 'TM' or 'TE' or 'both'

    return:
        2 x 2 complex matrix:
            element (tensor): 1 x number of frequencies x number of angles x number of pol
    '''
    kx = torch.sqrt(torch.pow(k * refractive_index, 2)  - torch.pow(ky, 2))

    TEpol = -torch.pow(refractive_index, 2)
    TMpol = torch.ones_like(TEpol)

    if pol == 'TM':
        pol_multiplier = TMpol
    elif pol == 'TE':
        pol_multiplier = TEpol
    else:
        pol_multiplier = torch.cat([TMpol, TEpol], dim = -1)

    return ((1., 0), (1., 0.), (-kx / k / pol_multiplier, 0.), (kx / k / pol_multiplier, 0.))


#def TMM_solver(thicknesses, refractive_indices, n_bot, n_top, k, theta, pol = 'TM'):
#GU5/9: modifiqué para considerar transmisión
def TMM_solver(self, thicknesses, refractive_indices, n_bot, n_top, k, theta, pol = 'TM'):
    '''
    args:
        thickness (tensor): batch size x number of layers
        refractive_indices (tensor): batch size x number of layers x (number of frequencies or 1)
        k (tensor): number of frequencies
        theta (tensor): number of angles
        n_bot (tensor): 1 or number of frequencies
        n_top (tensor): 1 or number of frequencies
        pol (str): 'TM' or 'TE' or 'both'
     
    return:
        2 x 2 complex matrix:
            element (tensor): batch size x number of frequencies x number of angles x number of pol
    '''
    # adjust the format
    n_bot = n_bot.view(1, -1, 1, 1)
    n_top = n_top.view(1, -1, 1, 1)
    k = k.view(1, -1, 1, 1)
    ky = k * n_bot * torch.sin(theta.view(1, 1, -1, 1))

    # transfer matrix calculation
    T_stack = transfer_matrix_stack(thicknesses, refractive_indices, k, ky, pol)
    
    # amplitude to field convertion
    A2F_bot = amp2field(n_bot, k, ky, pol)
    A2F_top = amp2field(n_top, k, ky, pol)
    
    # S matrix
    S_stack = matrix_mul(matrix_inv(A2F_top), matrix_mul(T_stack, A2F_bot))
    
    # reflection : |S21|² / |S22|²
    Reflection = torch.pow(complex_abs(S_stack[2]), 2) / torch.pow(complex_abs(S_stack[3]), 2)
    """
    # Transmission: |S12|² / |S22|² GU14/9:
    kx_top = torch.sqrt((k * n_top)**2 - ky**2)
    kx_bot = torch.sqrt((k * n_bot)**2 - ky**2)
    correction_factor = torch.real(n_top * kx_top) / torch.real(n_bot * kx_bot)
    Transmission = correction_factor * (torch.pow(complex_abs(S_stack[1]), 2) / torch.pow(complex_abs(S_stack[3]), 2))

    """
    #GU5/9: modifiqué para considerar 
    #GU5/9: modifiqué para considerar transmisión
    """
    # Calcular cosenos de los ángulos en las interfaces
    cos_theta_bot = torch.sqrt(1 - (ky / (k * n_bot))**2)
    cos_theta_top = torch.sqrt(1 - (ky / (k * n_top))**2)

    if pol == 'TM':
        impedance_ratio = (n_top * cos_theta_top) / (n_bot * cos_theta_bot)
    elif pol == 'TE':
        impedance_ratio = (n_bot * cos_theta_bot) / (n_top * cos_theta_top)
    else:
        # En caso de 'both', concatenar ambas polarizaciones
        imp_TM = (n_top * cos_theta_top) / (n_bot * cos_theta_bot)
        imp_TE = (n_bot * cos_theta_bot) / (n_top * cos_theta_top)
        impedance_ratio = torch.cat([imp_TM, imp_TE], dim=-1)

    # Transmitancia
    T22_abs2 = torch.pow(complex_abs(S_stack[3]), 2)
    Transmission = (1.0 / T22_abs2) * torch.real(impedance_ratio)
    """
      # Calcular cosenos de los ángulos en las interfaces
    cos_theta_bot = torch.sqrt(1 - (ky / (k * n_bot))**2)
    cos_theta_top = torch.sqrt(1 - (ky / (k * n_top))**2)

    # Calculamos el factor de corrección para la intensidad transmitida según la polarización
    if pol == 'TM':
        factor = torch.real((n_top * cos_theta_top) / (n_bot * cos_theta_bot))
    elif pol == 'TE':
        factor = torch.real((n_bot * cos_theta_top) / (n_top * cos_theta_bot))
    else:
        # En caso de 'both', concatenamos ambos factores de corrección
        factor_TM = torch.real((n_top * cos_theta_top) / (n_bot * cos_theta_bot))
        factor_TE = torch.real((n_bot * cos_theta_top) / (n_top * cos_theta_bot))
        factor = torch.cat([factor_TM, factor_TE], dim=-1)

    # Módulo cuadrado de S22 (coeficiente de transmisión)
    T22_abs2 = torch.pow(complex_abs(S_stack[3]), 2)

    # Transmitancia en intensidad
    Transmission = (1.0 / T22_abs2) * factor


    if self.spectra:        
        return Reflection
    else:     
        return Transmission
