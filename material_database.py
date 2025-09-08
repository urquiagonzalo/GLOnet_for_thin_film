import numpy as np
import pandas as pd
import torch

class MatDatabase(object):
	"""docstring for MatDatabase
		Parameters: 
			material_key: list of material names
	"""
	def __init__(self, material_key, database_folder): #GU8/9: modificado para considerar distintas carpetas desde el código principal. 
		super(MatDatabase, self).__init__()
		self.material_key = material_key
		self.num_materials = len(material_key)
		self.mat_database = self.build_database() #GU8/9: agregado para considerar distintas carpetas desde el código principal. 
		self.database_folder = database_folder

	def build_database(self):
		mat_database = {}
		
		#%% Read in the dispersion data of each material
		for i in range(self.num_materials):
			#GU8/9: modificado para considerar distintas carpetas desde el código principal. 
			#GU8/9: en el programa principal agregué params.database_folder = '...' 
			file_name = f'./{self.database_folder}/mat_{self.material_key[i]}.xlsx'
			
			try: 
				A = np.array(pd.read_excel(file_name))
				mat_database[self.material_key[i]] = (A[:, 0], A[:, 1], A[:, 2])
			except NameError:
				print('The material database does not contain', self.material_key[i])

		return mat_database


	def interp_wv(self, wv_in, material_key, ignoreloss = False):
		'''
			parameters
				wv_in (tensor) : number of wavelengths
				material_key (list) : number of materials

			return
				refractive indices (tensor or tuple of tensor) : number of materials x number of wavelengths
		'''
		n_data = np.zeros((len(material_key), wv_in.size(0)))
		k_data = np.zeros((len(material_key), wv_in.size(0)))
		for i in range(len(material_key)):
			mat = self.mat_database[material_key[i]]
			n_data[i, :] = np.interp(wv_in, mat[0], mat[1])
			k_data[i, :] = np.interp(wv_in, mat[0], mat[2])

		if ignoreloss:
			return torch.tensor(n_data)
		else:
			return (torch.tensor(n_data), torch.tensor(k_data))




		
