import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

def lse(x,axis=None) :
	if axis is None :
		x_max = np.max(x)
		return x_max + np.log(np.sum(np.exp(x-x_max)))
	else :
		x_max = np.max(x,axis=axis)
		return x_max + np.log(np.sum(np.exp(x-x_max[:,np.newaxis]),axis=axis))

def forwards_backwards(X,
	                   A,
	                   pi,
	                   emission,
	                   emission_params={},
	                   likelihood=False,
	                   return_normed=True,
	                   log_sum_exp=False) :
	# Assume inputs are np.arrays for now
	K = A.shape[0]
	assert K == len(pi)
	if type(X) == list :
		N = len(X)
		X = np.array(X)
	elif type(X) == np.ndarray :
		N = X.shape[0]
	
	emissions = np.zeros((N,K))
	for n in range(N) :
		for k in range(K) :
			#emissions[n,k] = emission_params[X[n,:],k]
			emissions[n,k] = emission(X[n,:],k,params=emission_params)
	
	alphas = np.zeros((N,K))
	betas  = np.zeros((N,K))
	betas[-1,:] = 1.

	#\alpha(z_1) = p(x_1,z_1) = p(x_1|z_1)p(z_1)
	for k in range(K) :
		if log_sum_exp :
			alphas[0,k] = pi[k] + emissions[0,k]
		else :
			alphas[0,k] = pi[k]*emissions[0,k]
	for na,nb in zip(range(1,N),reversed(range(N-1))) :
		for k in range(K) :
			if log_sum_exp :
				alphas[na,k] = emissions[na,k]+lse(alphas[na-1,:]+A[:,k])
				betas[nb,k] = lse(emissions[nb+1,:]+betas[nb+1,:]+A[k,:])
			else :
				alphas[na,k] = emissions[na,k]*np.sum(alphas[na-1,:]*A[:,k])
				betas[nb,k] = np.sum(emissions[nb+1,:]*betas[nb+1,:]*A[k,:])

	#print(emissions)
	#print(np.sum(alphas == 0,axis=1))
	#print(np.sum(betas == 0,axis=1))

	#\gamma(\bm{z}_i) = \frac{1}{p(\bm{X})}\alpha(\bm{z}_i)\beta(\bm{z}_i)
	#p(\bm{X}) = \sum_{\bm{z}_j\in Supp(\bm{z})}\alpha(\bm{z}_i)\beta(\bm{z}_i)
	#Note that p(\bm{X}) is also is a function of where you are on the chain (since it's effectively
	#normalizing the product \alpha(\bm{z}_i)\beta(\bm{z}_i)). So I find it easier to think of the following
	#p(\bm{X}) = Z(\bm{X},i)
	#\gamma(\bm{z}_i) = \frac{1}{Z(\bm{X},i)}\alpha(\bm{z}_i)\beta(\bm{z}_i)
	if log_sum_exp :
		gamma_no_norm = alphas + betas
	else :
		gamma_no_norm = alphas*betas
	if return_normed :
		if log_sum_exp :
			gammas = gamma_no_norm-lse(gamma_no_norm,axis=1)[:,np.newaxis]
		else :
			gammas = gamma_no_norm/np.sum(gamma_no_norm,axis=1)[:,np.newaxis]
	#\xi(\bm{z}_{i-1},\bm{z}_i) = \frac{1}{p(\bm{X})}\alpha(\bm{z}_{i-1})
	#\beta(\bm{z}_i)p(\bm{x}_i|\bm{z}_i)p(\bm{z}_i|\bm{z}_{i-1})
	xi_no_norm = np.zeros((N-1,K,K))
	if return_normed :
		xis = xi_no_norm.copy()
	for n in range(1,N) :
		for k in range(K) :
			if log_sum_exp :
				xi_no_norm[n-1,k,:] = alphas[n-1,k]+betas[n,:]+emissions[n,:]+A[k,:]
			else :
				xi_no_norm[n-1,k,:] = alphas[n-1,k]*betas[n,:]*emissions[n,:]*A[k,:]
		if return_normed :
			if log_sum_exp :
				xis[n-1,:,:] =  xi_no_norm[n-1,:,:]-lse(xi_no_norm[n-1,:,:])
			else :
				xis[n-1,:,:] =  xi_no_norm[n-1,:,:]/np.sum(xi_no_norm[n-1,:,:])
			#denominator = np.dot(np.dot(alpha[t, :].T, a) * b[:, V[t + 1]].T, beta[t + 1, :])
	#print(xis[5,:,:])
	#print(np.sum(xis,axis=(0,1)))
	#gammas = np.sum(xis,axis=1)
	#print(gammas[0,:])

	if likelihood :
		l = np.sum(alphas[-1,:])

	if not return_normed :
		if likelihood :
			return gamma_no_norm,xi_no_norm,l
		else :
			return gamma_no_norm,xi_no_norm,None
	else :
		if likelihood :
			return gammas,xis,l
		else :
			return gammas,xis,None

class HMM :
	def __init__(self,
		         A=None,
		         pi=None,
		         emissions=None,
		         update_initial_state=True) :
		self.A = A
		self.pi = pi
		self.emissions = emissions
		self.update_initial_state = update_initial_state

	def lse(self,x,axis=None) :
		if axis is None :
			x_max = np.max(x)
			return x_max + np.log(np.sum(np.exp(x-x_max)))
		else :
			x_max = np.max(x,axis=axis)
			return x_max + np.log(np.sum(np.exp(x-x_max),axis=axis))

	def fit(self,X,log_sum_exp=False,stopping_criterion={'num_iters':100}) :
		if 'num_iters' in stopping_criterion :
			for n in range(stopping_criterion['num_iters']) :
				gammas,xis,likelihood = forwards_backwards(X,
					                                       self.A,
					                                       self.pi,
					                                       self.emit,
					                                       emission_params={},
					                                       likelihood=True,
					                                       return_normed=True,
					                                       log_sum_exp=log_sum_exp)
				if self.update_initial_state :
					if log_sum_exp :
						self.pi = gammas[0,:]-self.lse(gammas[0,:])
					else :
						self.pi = gammas[0,:]/np.sum(gammas[0,:])
				if log_sum_exp :
					self.A = self.lse(xis,axis=0)-(self.lse(xis,axis=(0,2))[:,np.newaxis])
				else :
					self.A = np.sum(xis,axis=0)/(np.sum(xis,axis=(0,2))[:,np.newaxis])
				self.update_emissions(**{'X':X,'gammas':gammas,'xis':xis,'log_sum_exp':log_sum_exp})
				

	def emit(self,x,z,params={}) :
		pass

	def update_emissions(self,**kwargs) :
		pass

class CategoricalHMM(HMM) :
	def __init__(self,
		         A=None,
		         pi=None,
		         emissions=None,
		         update_initial_state=True) :
		super(CategoricalHMM,self).__init__(A=A,
			                                pi=pi,
			                                emissions=emissions,
			                                update_initial_state=update_initial_state)

	def emit(self,x,z,params={}) :
		return self.emissions[x,z]

	def update_emissions(self,**kwargs) :
		log_sum_exp = kwargs.get('log_sum_exp',False)
		print(log_sum_exp)
		X = kwargs['X']
		gammas = kwargs['gammas']
		for i in range(self.emissions.shape[0]) :
			mask = np.squeeze(X) == i
			if log_sum_exp :
				self.emissions[i,:] = self.lse(gammas[mask,:],axis=0)-\
			                          self.lse(gammas,axis=0)
			else :
				self.emissions[i,:] = np.sum(gammas[mask,:],axis=0)/\
			                          np.sum(gammas,axis=0)

class GaussianHMM(HMM) :
	def __init__(self,
		         A=None,
		         pi=None,
		         emissions=None,
		         update_initial_state=True) :
		super(GaussianHMM,self).__init__(A=A,
			                             pi=pi,
			                             emissions=emissions,
			                             update_initial_state=update_initial_state)

	def emit(self,x,z,params={}) :
		"""
		Gaussian Emissions x_i|z_i ~ N(mu_{z_i},sigma_{z_i})
		self.emissions['means'].shape = (K,D)
		self.emissions['covs'].shape  = (K,D,D)
		where self.emissions['means'][k,:] = mu_k
		and self.emissions['covs'][k,:,:] = cov_k
		"""
		return multivariate_normal.pdf(x,
			                           mean=self.emissions['means'][z,:],
			                           cov=self.emissions['covs'][z,:,:])

	def update_emissions(self,**kwargs) :
		X = kwargs['X']
		gammas = kwargs['gammas']
		N = X.shape[0]
		D = X.shape[1]
		K = gammas.shape[1]
		#print(gammas)
		#(K,N)*(N,D)=(K,D)
		#self.emissions['means'] = gammas.T.dot(X)/np.sum(gammas,axis=0)[:,np.newaxis]
		for k in range(K) :
			mu_totals = np.zeros(D,)
			for n in range(N) :
				mu_totals += gammas[n,k]*X[n,:]
			mu_totals /= np.sum(gammas[:,k])
			self.emissions['means'][k,:] = mu_totals
		for k in range(K) :
			cov_totals = np.zeros((D,D))
			for n in range(N) :
				x_mu = X[n,:] - self.emissions['means'][k,:]
				outer_prod = np.outer(x_mu,x_mu)
				outer_prod *= gammas[n,k]
				cov_totals += outer_prod
			cov_totals /= np.sum(gammas[:,k])
			self.emissions['covs'][k,:,:] = cov_totals
		"""
		for k in range(K) :
			x_mu = X - self.emissions['means'][k,:] #(N,D) - (D,) = (N,D)
			outer_prod = (gammas[:,k]*x_mu.T).dot(x_mu)
			self.emissions['covs'][k,:,:] = outer_prod/np.sum(gammas[:,k])
			#self.emissions['covs'][k,:,:] = np.einsum('nd,nj->dj',gammas[:,k,np.newaxis]*x_mu,x_mu)/np.sum(gammas[:,k])
		"""

def main() :
	#"""
	A = np.ones((2, 2))
	A = A / np.sum(A, axis=1)
	b = np.array(((1, 3, 5), (2, 4, 6)))
	b = b / np.sum(b, axis=1).reshape((-1, 1))
	b = b.T
	X = pd.read_csv('/home/chrism/sw/hmm/data/test.csv')
	X = X['Visible'].values[:,np.newaxis]

	c_hmm = CategoricalHMM(A=A,pi=np.array([0.5,0.5]),emissions=b,update_initial_state=False)
	c_hmm.fit(X,log_sum_exp=True)
	print(c_hmm.A)
	print(np.exp(c_hmm.A))
	"""
	A = np.ones((3,3))
	A /= 3
	b = {'means':np.array([[0,0],[5,5],[-5,5]],dtype=np.float64),
	     'covs':np.array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]],dtype=np.float64)}
	#X1 = np.random.multivariate_normal(mean=b['means'][0,:],cov=b['covs'][0,:,:],size=300)
	#X2 = np.random.multivariate_normal(mean=b['means'][1,:],cov=b['covs'][1,:,:],size=300)
	#X3 = np.random.multivariate_normal(mean=b['means'][2,:],cov=b['covs'][2,:,:],size=300)
	#X = np.concatenate((X1,X2,X3))
	X = np.random.multivariate_normal(mean=np.array([0,0]),cov=np.array([[5,0],[0,5]]),size=900)
	g_hmm = GaussianHMM(A=A,pi=np.array([0.5,0.5,0.5]),emissions=b,update_initial_state=False)
	g_hmm.fit(X)
	print(g_hmm.A)
	#"""

if __name__ == '__main__' :
	main()