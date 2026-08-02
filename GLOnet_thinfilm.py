import matplotlib.pyplot as plt
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import UnivariateSpline
from TMM import *
from tqdm import tqdm
from net import Generator, ResGenerator

class GLOnet():
    def __init__(self, params):
        # GPU 
        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
            
        # construct
        if params.net == 'Res':
            self.generator = ResGenerator(params)
        else:
            self.generator = Generator(params)
        
        if self.cuda: 
            self.generator.cuda()
        self.optimizer = torch.optim.Adam(self.generator.parameters(), lr=params.lr, betas = (params.beta1, params.beta2), weight_decay = params.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = params.step_size, gamma = params.step_size)
        
        # training parameters
        self.noise_dim = params.noise_dim
        self.numIter = params.numIter
        self.batch_size = params.batch_size
        self.sigma = params.sigma
        self.alpha_sup = params.alpha_sup
        self.iter0 = 0
        self.alpha = 0.1
        
        # simulation parameters
        self.user_define = params.user_define
        
        # True en programa principal selecciona el modo sensor con métrica modificada y False selecciona optimizador clásico con la loss clásica 
        if params.sensor: 
            self.sensor = True
        else:
            self.sensor = False
        
        #GU5/9: True en programa principal considera refelexión o y False transmisión 
        self.spectra = params.spectra 
        
        if params.user_define:
            self.n_database = params.n_database
        else:
            self.materials = params.materials
            self.matdatabase = params.matdatabase

        self.n_bot = params.n_bot.type(self.dtype)  # number of frequencies or 1
        self.n_top = params.n_top.type(self.dtype)  # number of frequencies or 1
        self.k = params.k.type(self.dtype)  # number of frequencies
        self.theta = params.theta.type(self.dtype) # number of angles
        self.pol = params.pol # str of pol
        self.target_spectra = params.target_spectra.type(self.dtype) # cambié "reflection" por "spectra" ... self.target_reflection = params.target_reflection.type(self.dtype) 
        # 1 x number of frequencies x number of angles x (number of pol or 1)

        # Si trabajamos en modo sensor, leemos los archivos CSV del LED y del LDR
        if self.sensor: 
            self.led_spline = self._create_spline("true-green-osram.csv")
            self.ldr_spline = self._create_spline("ldr.csv")
        
        # tranining history
        self.loss_training = []
        self.refractive_indices_training = []
        self.thicknesses_training = []
        self.mse_training = []                                       #GU: mse
        self.batch_mse_training = []                                 #GU: mse batch

    # Función definida para interpolar los datos de los archivos CSV del LED y del LDR
    def _create_spline(self, filename):
        df = pd.read_csv(filename, sep=';', decimal=',')
        df.columns = ['Wavelength [nm]', 'Reflection spectra']
        spline = UnivariateSpline(df['Wavelength [nm]'] / 1000, df['Reflection spectra'])
        spline.set_smoothing_factor(0.006)
        return spline
    
        
    def train(self,seed):
        self.generator.train()

        # GU7/12: Lista de iteraciones que querés descargar
        self.iters_to_download = [400]
            
        # training loop
        with tqdm(total=self.numIter) as t:
            it = self.iter0  
            while True:
                it +=1 

                # normalized iteration number
                normIter = it / self.numIter

                # discretizaton coeff.
                self.update_alpha(normIter)
                
                # terminate the loop
                if it > self.numIter:
                    return 

                # sample z
                z = self.sample_z(self.batch_size)

                # generate a batch of iamges
                # if self.sensor guarda espesores e indices de refracciones para las matrices densas y porosas
                if self.sensor: 
                    thicknesses, refractive_indices_air, refractive_indices_water, P = self.generator(z, self.alpha)
                    #thicknesses, refractive_indices_air, refractive_indices_water, _ = self.generator(z, self.alpha)
                else:    
                    thicknesses, refractive_indices, P = self.generator(z, self.alpha) #
                
                # ---------------------------------------------------------
                # AGREGADO PARA GUARDAR ESTRUCTURAS EN ALGUNAS ITERACIONES
                # ---------------------------------------------------------

                # 1️⃣ Guardar espesores
                thicknesses_np = thicknesses.detach().cpu().numpy()
                np.savetxt(f"Espesores_iter_{it}_Semilla{seed}.txt",
                thicknesses_np * 1000, fmt="%.6f")
                
                # 2️⃣ Guardar índices de refracción (aplanado a 2D)
                refidx_np = refractive_indices.detach().cpu().numpy()
                refidx_flat = refidx_np.reshape(-1, refidx_np.shape[2])
                # np.savetxt(f"refidx_iter_{it}_Semilla{seed}.txt",
                           #refidx_flat, fmt="%.6f")
                
                #3️⃣ Guardar nombres de materiales por capa
                result_mat = torch.argmax(P, dim=2).detach().cpu().numpy()
                with open(f"Materiales_iter_{it}_Semilla{seed}.txt", 'w') as f:
                    for row in result_mat:
                        f.write(','.join([self.materials[i] for i in row]) + '\n')
                        
                # ============================================================
                #        DESCARGAR SÓLO ALGUNAS ITERACIONES DEFINIDAS
                # ============================================================
                if it in self.iters_to_download:
                    from google.colab import files
                    files.download(f"Espesores_iter_{it}_Semilla{seed}.txt")
                    #files.download(f"refidx_iter_{it}_Semilla{seed}.txt")
                    files.download(f"Materiales_iter_{it}_Semilla{seed}.txt")
                    
                ################################################################################    
                ## if it == self.numIter:   # última iteración
                    # 1️⃣ Guardar espesores
                    #thicknesses_last = thicknesses.detach().cpu().numpy()
                    #np.savetxt(f"Espesores_Ultima_iter_{self.numIter}_Semilla{seed}.txt",
                               #thicknesses_last * 1000, fmt="%.6f")                
                
                    # 2️⃣ Guardar índices de refracción (aplanado a 2D)
                    #refidx_last = refractive_indices.detach().cpu().numpy()
                    # aplanamos batch x capas como filas, frecuencias como columnas
                    #refidx_flat = refidx_last.reshape(-1, refidx_last.shape[2])
                    #np.savetxt(f"refidx_last_iter_{self.numIter}.txt", refidx_flat, fmt="%.6f")
                
                    # 3️⃣ Guardar nombres de materiales por capa
                    # Convertimos el tercer valor devuelto por el generador (antes '_') a P
                    #_, _, P = self.generator(z, self.alpha)
                    #result_mat = torch.argmax(P, dim=2)  # batch x num_layers
                    #result_mat_np = result_mat.detach().cpu().numpy()
                    
                    #with open(f"Materiales_Ultima_iter_{self.numIter}_Semilla{seed}.txt", 'w') as f:
                    #    for row in result_mat_np:  # cada fila = un diseño
                    #        f.write(','.join([self.materials[i] for i in row]) + '\n')
                    #######################################################################
                    
                    # =================================================================== #
                    # GUARDAR SOLO EL MSE DEL ÚLTIMO CONJUNTO DE BATCH - ULTIMA ITERACIÓN #
                    # =================================================================== #
                    #last_batch_mse = mse_per_sample.detach().cpu().numpy()
                    #np.savetxt(f"msexbatch_last_iter_{self.numIter}.txt",
                    #           last_batch_mse, fmt="%.8f")
                   
                    # si estás en Colab:
                    #from google.colab import files
                    #files.download(f"Espesores_Ultima_iter_{self.numIter}_Semilla{seed}.txt")
                    #files.download(f"refidx_last_iter_{self.numIter}_Semilla{seed}.txt")
                    #files.download(f"Materiales_Ultima_iter_{self.numIter}_Semilla{seed}.txt")
                    #files.download(f"msexbatch_last_iter_{self.numIter}_Semilla{seed}.txt")
                 # -----------------------------------------------
                 # -----------------------------------------------

                # calculate efficiencies and gradients using EM solver
                #GU5/9: modificado para considerar refelexión (True en programa principal) o transmisión (False) 
                if self.spectra:
                    if self.sensor:
                        reflection_air = TMM_solver(self, thicknesses,refractive_indices_air, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                        reflection_water = TMM_solver(self, thicknesses,refractive_indices_water, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                    else: # optimizador clásico
                        reflection = TMM_solver(self, thicknesses, refractive_indices, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                else:    
                    if self.sensor:
                        transmission_air = TMM_solver(self, thicknesses,refractive_indices_air, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                        transmission_water = TMM_solver(self, thicknesses,refractive_indices_water, self.n_bot, self.n_top, self.k, self.theta, self.pol)
                    else: # optimizador clásico
                        transmission = TMM_solver(self, thicknesses, refractive_indices, self.n_bot, self.n_top, self.k, self.theta, self.pol) #GU5/9: agrego transmisión        
               
                # GU5/9: podrían ser las dos (VER)
                # reflection, transmission = TMM_solver(thicknesses, refractive_indices, self.n_bot, self.n_top, self.k, self.theta, self.pol)

                # free optimizer buffer 
                self.optimizer.zero_grad()

                # construct the loss 
                #GU5/9: modificado para considerar refelexión (True en programa principal) o transmisión (False) 
                difrel_Obj = 1
                if self.spectra:
                    sensor_signal = self.sensor_signal_1(self.k, reflection_air, reflection_water) if self.sensor else None                                   # Se usa en modo sensor 
                    #sensor_signal = self.sensor_signal_2(self.k, reflection_empty, reflection_full_A, reflection_full_B) if self.sensor else None            # 2 materiales
                    
                    g_loss = self.global_loss_function(sensor_signal) if self.sensor else self.global_loss_function(reflection)        # En modo sensor usa "sensor_signal" y en modo clásico "reflection"
                    g_mse = self.global_mse_function(sensor_signal) if self.sensor else self.global_mse_function(reflection)           #GU: mse
                    mse_per_sample = self.batch_mse_function(sensor_signal) if self.sensor else self.batch_mse_function(reflection)    #GU: mse batch 
                    
                    FM = torch.pow(sensor_signal - difrel_Obj, 2) if self.sensor else torch.pow(reflection - self.target_spectra, 2)         # VER AGREGADO DE DONDE VIENE
                
                else:
                    sensor_signal = self.sensor_signal_1(self.k, transmission_air, transmission_water) if self.sensor else None                                # métrica para usar en sensor
                    # sensor_signal = self.sensor_signal_2(self.k, transmission_empty, transmission_full_A, transmission_full_B) if self.sensor else None      # 2 materiales
                    
                    g_loss = self.global_loss_function(sensor_signal) if self.sensor else self.global_loss_function(transmission)   
                    g_mse = self.global_mse_function(sensor_signal) if self.sensor else self.global_mse_function(transmission)           #GU: mse
                    mse_per_sample = self.batch_mse_function(sensor_signal) if self.sensor else self.batch_mse_function(transmission)    #GU: mse batch 

                    FM = torch.pow(sensor_signal - difrel_Obj, 2) if self.sensor else torch.pow(transmission - self.target_spectra, 2)         # VER AGREGADO DE DONDE VIENE
                              
                # record history
                #self.record_history(g_loss, thicknesses, refractive_indices,g_mse)                  #GU: solo mse
                self.record_history(g_loss, thicknesses, refractive_indices,g_mse, mse_per_sample)   #GU: mse y mse por batch
                
                # train the generator
                g_loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                
                # update progress bar
                t.update()
    
    def evaluate(self, num_devices, kvector = None, inc_angles = None, pol = None, grayscale=True):
        if kvector is None:
            kvector = self.k
        if inc_angles is None:
            inc_angles = self.theta
        if pol is None:
            pol = self.pol            

        self.generator.eval()
        z = self.sample_z(num_devices) # Llama al creador de números aleatorios

        # IF (self.sensor) PARA CONSIDERAR CASO DE SENSOR
        if self.sensor:
            thicknesses, refractive_indices_air, refractive_indices_water, P = self.generator(z, self.alpha)
            result_mat = torch.argmax(P, dim=2).detach() # batch size x number of layer
            
            if not grayscale:
                ref_idx_air, ref_idx_water  = self._calculate_refractive_indices(kvector)     # calculate_refractive_indices es una función definida más abajo
            else:
                if self.user_define:
                    ref_idx_air, ref_idx_water = refractive_indices_air, refractive_indices_water
                else:   
                    n_database_air = self.to_cuda_if_available(self.matdatabase_air.interp_wv(2 * math.pi/kvector, self.materials_air, True).unsqueeze(0).unsqueeze(0))          # lee n
                    #n_database_air = self.to_cuda_if_available(self.matdatabase_air.interp_wv(2 * math.pi/kvector, self.materials_air, False).unsqueeze(0).unsqueeze(0))        # lee n y k
                    ref_idx_air = torch.sum(P.unsqueeze(-1) * n_database_air, dim=2)
                    
                    n_database_water = self.to_cuda_if_available(self.matdatabase_water.interp_wv(2 * math.pi/kvector, self.materials_water, True).unsqueeze(0).unsqueeze(0))    # lee n
                    #n_database_water = self.to_cuda_if_available(self.matdatabase_water.interp_wv(2 * math.pi/kvector, self.materials_water, False).unsqueeze(0).unsqueeze(0))  # lee n y k
                    ref_idx_full_water = torch.sum(P.unsqueeze(-1) * n_database_water, dim=2)
 
                    # Modificado para considerar refelexión (True en programa principal) o transmisión (False) 
                    if self.spectra:
                        reflection_air   = TMM_solver(thicknesses, ref_idx_air, self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
                        reflection_water = TMM_solver(thicknesses, ref_idx_water, self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
                        
                        sensor_signal = self.sensor_signal_1(self.to_cuda_if_available(kvector), reflection_air, reflection_water)
                        #sensor_signal = self.sensor_signal_2(self.to_cuda_if_available(kvector), reflection_empty, reflection_full_A, reflection_full_B)      # 2 materiales distintos
                        return (thicknesses, result_mat, sensor_signal, ref_idx_air, reflection_air, ref_idx_water, reflection_water)
                    else:
                        transmission_air   = TMM_solver(thicknesses, ref_idx_air  , self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
                        transmission_water = TMM_solver(thicknesses, ref_idx_water, self.n_bot, self.n_top, self.to_cuda_if_available(kvector), self.to_cuda_if_available(inc_angles), pol)
                        
                        sensor_signal = self.sensor_signal_1(self.to_cuda_if_available(kvector), transmission_air, transmission_water)
                        #sensor_signal = self.sensor_signal_2(self.to_cuda_if_available(kvector), transmission_empty, transmission_full_A, transmission_full_B) # 2 materiales distintos
                        return (thicknesses, result_mat, sensor_signal, ref_idx_air, transmission_air, ref_idx_water, transmission_water)
        
        # ELSE PARA TRABAJAR CON LA VERSIÓN ORIGINAL
        else:
            thicknesses, refractive_indices, P = self.generator(z, self.alpha)
            result_mat = torch.argmax(P, dim=2).detach() # batch size x number of layer
            
            if not grayscale:     
                if self.user_define:
                    n_database = self.n_database # do not support dispersion    
                else:
                    n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, True).unsqueeze(0).unsqueeze(0).type(self.dtype)      # lee n
                    
                    #n_database = self.matdatabase.interp_wv(2 * math.pi / kvector,self.materials,False)                                                 # con lectura n y k (ver la idea)                 
                    #if isinstance(n_database, tuple): 
                    #    n_database = torch.stack(n_database, dim=-1)
                    #n_database_complex = torch.complex(n_database[..., 0],n_database[..., 1]).to(P.device)                                              
             
                one_hot = torch.eye(len(self.materials)).type(self.dtype)
                ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database, dim=2)
                #ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database_complex, dim=2)                                                      # con lectura n y k 
            else:
                if self.user_define:
                    ref_idx = refractive_indices
                else:
                    n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, True).unsqueeze(0).unsqueeze(0).type(self.dtype)      # lee n
                    
                    #n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, False)                                                 # con lectura n y k (ver la idea)  
                    #if isinstance(n_database, tuple):
                    #    n_database = torch.stack(n_database, dim=-1)
                    #n_database_complex = torch.complex(n_database[..., 0],n_database[..., 1]).to(P.device)                                           

                    ref_idx = torch.sum(P.unsqueeze(-1) * n_database, dim=2)
                    #ref_idx = torch.sum(P.unsqueeze(-1) * n_database_complex, dim=2)                                                                    # con lectura n y k  
        
            #GU5/9: modificado para considerar refelexión (True en programa principal) o transmisión (False) 
            if self.spectra:
                reflection = TMM_solver(self, thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
                return (thicknesses, ref_idx, result_mat, reflection)
            else:
                transmission = TMM_solver(self, thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
                return (thicknesses, ref_idx, result_mat, transmission)

    # Función extra
    def _calculate_refractive_indices(self, result_mat, kvector):
        if self.user_define:
            n_database_air = self.to_cuda_if_available(self.n_database_air)      # do not support dispersion
            n_database_water = self.to_cuda_if_available(self.n_database_water)  # do not support dispersion
        else:
            n_database_air  = self.to_cuda_if_available(self.matdatabase_air.interp_wv(2 * math.pi / kvector, self.materials_empty, False).unsqueeze(0).unsqueeze(0))
            n_database_water = self.to_cuda_if_available(self.matdatabase_water.interp_wv(2 * math.pi / kvector, self.materials_full_A, False).unsqueeze(0).unsqueeze(0))
        one_hot = self.to_cuda_if_available(torch.eye(len(self.materials_empty)))
        one_hot_mat = one_hot[result_mat].unsqueeze(-1)
        ref_idx_air   = torch.sum(one_hot_mat * n_database_air, dim=2)
        ref_idx_water = torch.sum(one_hot_mat * n_database_water, dim=2)
        return (ref_idx_air, ref_idx_water) 
    
    def _TMM_solver(self, thicknesses, result_mat, kvector = None, inc_angles = None, pol = None):
        if self.sensor:
            if kvector is None:
                kvector = self.k
            if inc_angles is None:
                inc_angles = self.theta
            if pol is None:
                pol = self.pol  
            n_database_air = self.matdatabase_air.interp_wv(2 * math.pi/kvector, self.materials_air, False).unsqueeze(0).unsqueeze(0)
            n_database_water = self.matdatabase_water.interp_wv(2 * math.pi/kvector, self.materials_water, False).unsqueeze(0).unsqueeze(0)
            
            one_hot = torch.eye(len(self.materials_empty))
            one_hot_mat = one_hot[result_mat].unsqueeze(-1)
            
            ref_idx_air   = torch.sum(one_hot_mat * n_database_air, dim=2)
            ref_idx_water = torch.sum(one_hot_mat * n_database_water, dim=2)

            if self.spectra:
                reflection_air     = TMM_solver(thicknesses, ref_idx_air, self.n_bot, self.n_top, kvector, inc_angles, pol)
                reflection_water   = TMM_solver(thicknesses, ref_idx_water, self.n_bot, self.n_top, kvector, inc_angles, pol)
                return (reflection_air, reflection_water) 
            else:
                transmission_air   = TMM_solver(thicknesses, ref_idx_air, self.n_bot, self.n_top, kvector, inc_angles, pol)
                transmission_water = TMM_solver(thicknesses, ref_idx_water, self.n_bot, self.n_top, kvector, inc_angles, pol)
                return (transmission_air, transmission_water)

        # ELSE PARA TRABAJAR CON LA VERSIÓN ORIGINAL
        else:
            if kvector is None:
                kvector = self.k
            if inc_angles is None:
                inc_angles = self.theta
            if pol is None:
                pol = self.pol  
            n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, True).unsqueeze(0).unsqueeze(0).type(self.dtype)              # lee n
            
            #n_database = self.matdatabase.interp_wv(2 * math.pi/kvector, self.materials, False)                                                         # con lectura n y k (ver la idea)  
            #if isinstance(n_database, tuple):
            #    n_database = torch.stack(n_database, dim=-1)
            #n_database_complex = torch.complex(n_database[..., 0],n_database[..., 1]).to(P.device)                                                      
            
            one_hot = torch.eye(len(self.materials)).type(self.dtype)
            ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database, dim=2)
            #ref_idx = torch.sum(one_hot[result_mat].unsqueeze(-1) * n_database_complex, dim=2)
            
            #reflection = TMM_solver(thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
            #return reflection
            #GU5/9: modificado para considerar refelexión (True en programa principal) o transmisión (False) 
            if self.spectra:
                reflection = TMM_solver(self, thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
                return reflection
            else: 
                transmission = TMM_solver(self, thicknesses, ref_idx, self.n_bot, self.n_top, kvector.type(self.dtype), inc_angles.type(self.dtype), pol)
                return transmission

    def update_alpha(self, normIter):
        self.alpha = round(normIter/0.05) * self.alpha_sup + 1.

    # crea un tensor con la forma indicada, donde cada elemento es un número aleatorio tomado de una distribución normal estándar (media = 0, desviación estándar = 1).
    # Es decir, los números no son uniformes ni enteros; son valores continuos centrados en 0.
    def sample_z(self, batch_size):
        return (torch.randn(batch_size, self.noise_dim, requires_grad=True)).type(self.dtype)

   
    # Función para calcular la integral numérica de un espectro usando la regla del trapecio.
    def spectra_int(self, spectra, k, dim):
        lambdas = 2*math.pi/self.k
        return torch.trapz(spectra, lambdas, dim= dim)
        
    # Función que mide cuánta diferencia detecta un sensor entre dos espectros (vacío vs lleno), ponderado por la respuesta del sistema óptico
    def sensor_signal_1(self, k, spectra_empty, spectra_full): # función de la página 56
        lambdas = 2 * math.pi / self.k
        # La siguiente línea construye la respuesta espectral combinada del sistema (fuente (led)  + detector (ldr)) y la prepara como tensor en PyTorch. "R(λ)=LED(λ)⋅LDR(λ)"       
        led_x_ldr = self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas) * self.ldr_spline(lambdas)))   

        # Usamos "torch.diag(led_x_ldr)" que convierte el vector led_x_ldr en una matriz diagonal, donde wi=LED(λi)⋅LDR(λi)
        # matmul hace: [S1​,S2​,…,Sn​]⋅diag(w1​,w2​,…,wn​)=[S1​w1​,S2​w2​,…,Sn​wn​] # ver si da lo mismo usando "signal_empty = spectra_empty.squeeze() * led_x_ldr" debería ser más rápido

        signal_empty = torch.matmul(spectra_empty.squeeze(),torch.diag(led_x_ldr))                                         # Aplica la respuesta espectral del sistema (LED × LDR) al espectro spectra_empty, ponderando cada longitud de onda.
        signal_full = torch.matmul(spectra_full.squeeze(),torch.diag(led_x_ldr))                                           # Aplica la respuesta espectral del sistema (LED × LDR) al espectro spectra_full, ponderando cada longitud de onda.
        signal_diff = signal_empty - signal_full                                                                           # Diferencia  
        int_led = self.spectra_int(self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas))), self.k, dim = 0) # Calcula la integral del espectro del LED
        int_diff = self.spectra_int(signal_diff, self.k, dim = 1)                                                          # Calcula la integral de la diferencia de los espectros
        sensor_signal= torch.abs(int_diff)/int_led
        return sensor_signal  

    # Esta segunda función ya no mide una sola diferencia entre dos estados, sino que construye una métrica no lineal entre tres espectros distintos (A, B y vacío).
    def sensor_signal_2(self, k, spectra_empty, spectra_full_A, spectra_full_B):
        lambdas = 2 * math.pi / self.k
        led_x_ldr = self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas) * self.ldr_spline(lambdas)))
        int_led = self.spectra_int(self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas))), self.k, dim = 0)
        signal_empty = torch.matmul(spectra_empty.squeeze(),torch.diag(led_x_ldr))
        signal_empty_int = self.spectra_int(signal_empty, self.k, dim = 1)
        signal_A = torch.matmul(spectra_full_A.squeeze(),torch.diag(led_x_ldr))
        signal_A_int = self.spectra_int(signal_A, self.k, dim = 1)
        if torch.all(spectra_full_B == 1):
            print("Warning: spectra_full_B is all ones, using signal_empty for sensor_signal_2")
            signal_diff = signal_empty_int - signal_A_int  # Igual a sensor_signal_1 en este caso
        else:
            signal_B = torch.matmul(spectra_full_B.squeeze(),torch.diag(led_x_ldr))
            signal_B_int = self.spectra_int(signal_B, self.k, dim = 1)
            signal_diff = (signal_empty_int - signal_A_int) * (signal_A_int - signal_B_int) * (signal_B_int - signal_empty_int) / (int_led **3)
        #int_led = self.spectra_int(self.to_cuda_if_available(torch.from_numpy(self.led_spline(lambdas))), self.k, dim = 0)
        #int_diff = self.spectra_int(signal_diff, self.k, dim = 1)
        #sensor_signal= torch.abs(signal_diff)/int_led
        sensor_signal= torch.abs(signal_diff)
        return sensor_signal 
   
    def global_mse_function(self, reflection):                               #MSE GLOBAL (promedio de todos los MSE de cada batch)
        return torch.mean(torch.pow(reflection - self.target_spectra, 2)) 
        
    def batch_mse_function(self, reflection):                                #MSE DE CADA BATCH
        return torch.mean(torch.pow(reflection - self.target_spectra, 2), dim=(1,2,3))
        
    def global_loss_function(self, reflection): 
        if self.sensor:
            difrel_Obj = 1
            -torch.mean(torch.exp(-torch.pow(sensor_signal - difrel_Obj, 2)/self.sigma)) # 
        else:
            return -torch.mean(torch.exp(-torch.mean(torch.pow(reflection - self.target_spectra, 2), dim=(1,2,3))/self.sigma)) 
        
    def global_loss_function_robust(self, reflection, thicknesses):
        metric = torch.mean(torch.pow(reflection - self.target_spectra, 2), dim=(1,2,3))
        dmdt = torch.autograd.grad(metric.mean(), thicknesses, create_graph=True)
        return -torch.mean(torch.exp((-metric - self.robust_coeff *torch.mean(torch.abs(dmdt[0]), dim=1))/self.sigma))

    # def record_history(self, loss, thicknesses, refractive_indices,mse):                      #GU: solo mse
    def record_history(self, loss, thicknesses, refractive_indices,mse, mse_per_sample):        #GU: mse - batch
        self.loss_training.append(loss.detach())
        self.thicknesses_training.append(thicknesses.mean().detach())
        self.refractive_indices_training.append(refractive_indices.mean().detach())
        self.mse_training.append(mse.detach().item())                                        #GU: mse                                
        self.batch_mse_training.append(mse_per_sample.detach().cpu().numpy())                #GU: mse - batch
        
    def viz_training(self,seed): 
        #plt.figure(figsize = (20, 5))
        #plt.subplot(131)
        plt.plot(self.loss_training)
        #plt.plot(self.mse_training , color='orange')  # GU: grafico MSE
        #plt.ylabel('Loss', fontsize=18)
        #plt.xlabel('Iterations', fontsize=18)
        #plt.xticks(fontsize=14)
        #plt.yticks(fontsize=14)
        from google.colab import files
        with open(f"loss{seed}.txt", 'w') as f:
            f.write(', '.join([f"{x:.8f}" for x in self.loss_training]) + '\n\n')
            files.download(f"loss{seed}.txt")
        with open(f"msexbatch{seed}.txt", 'w') as f:
            for batch_mse in self.batch_mse_training:  # cada batch_mse es un np.array de tamaño batch_size
                f.write(', '.join([f"{x:.8f}" for x in batch_mse]) + '\n')  # una línea por batch
        files.download(f"msexbatch{seed}.txt") 
        with open(f"mse{seed}.txt", 'w') as f:                                      # Para guardar el MSE global (promedio de los todos los MSExbatch)
            f.write(', '.join([f"{x:.8f}" for x in self.mse_training]) + '\n\n')
            files.download(f"mse{seed}.txt") 

#Cada línea de sexbatch(seed).txt corresponde a una iteración. Cada línea tiene el mse correspondiente a cada batch. Si tengo 150 de bacth habrá 150 números. 
#El promedio de todas estos números se guarda en el archivo mse(seed).txt. Este archivo tiene 400 números. cada número es el promedio de cada iteración. 
#Es decir, el promedio de la primera línea de msexbatch1.txt corresponde al primer número que aparece en mse1.txt 
