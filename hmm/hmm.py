import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

class HMM :
	def __init__(self,
		         A=None,
		         pi=None,
		         emissions=None) :
		self.A = A
		self.pi = pi
		self.emissions = emissions
		self.alphas = None

	def lse(self,x,axis=None) :
		if axis is None :
			x_max = np.max(x)
			return x_max + np.log(np.sum(np.exp(x-x_max)))
		elif axis == 1 :
			x_max = np.max(x,axis=axis)
			return x_max + np.log(np.sum(np.exp(x-x_max[:,np.newaxis]),axis=axis))
		elif axis == 0 :
			x_max = np.max(x,axis=axis)
			return x_max + np.log(np.sum(np.exp(x-x_max),axis=axis))

	def fit(self,
		    X,
		    log_space=False,
		    update_pi=True,
		    stopping_criterion={'num_iters':100},
		    emission_params={}) :
		old_likelihood = -np.inf
		if 'num_iters' in stopping_criterion :
			for n in range(stopping_criterion['num_iters']) :
				gammas,xis,ll = self.forwards_backwards(X,
					                                    emission_params=emission_params,
					                                    log_likelihood=True,
					                                    return_normed=True,
					                                    log_space=log_space,
					                                    update_pi=update_pi)
				
				"""
				if np.log(likelihood) - old_likelihood < 1e-4 :
					print(f'Breaking after the {n}th iteration')
					break
				else :
					old_likelihood = np.log(likelihood)
				"""
	def predictive_distribution(self,xnew,emission_params={},log_space=False) :
		if self.alphas is None :
			raise
		K = self.A.shape[0]
		numerator = 0.
		for k in range(K) :
			transition = 0.
			for j in range(K) :
				transition += self.A[j,k]*self.alphas[-1,j]
			numerator += self.emit(xnew,k,params=emission_params)*transition
		return numerator/np.sum(self.alphas[-1,:])

	def posterior_distribution(self,log_space=False) :
		K = self.A.shape[0]
		posterior = np.zeros(K,)
		if log_space :
			denom = self.lse(self.alphas[-1,:])
		else :
			denom = np.sum(self.alphas[-1,:])
		for k in range(K) :
			if log_space :
				posterior[k] = self.lse(self.A[:,k] + self.alphas[-1,:])
			else :
				posterior[k] = np.sum(self.A[:,k]*self.alphas[-1,:])
		if log_space :
			return posterior - denom
		else :
			return posterior/denom

	def forwards_backwards(self,
		                   X,
	                       emission_params={},
	                       log_likelihood=True,
	                       return_normed=True,
	                       log_space=False,
	                       update_pi=True) :
		N = X.shape[0]
		K = self.A.shape[0]
		assert K == len(self.pi)
		
		emissions = np.zeros((N,K))
		for n in range(N) :
			for k in range(K) :
				emissions[n,k] = self.emit(X[n,:],k,params=emission_params)
		
		alphas = np.zeros((N,K))
		betas  = np.zeros((N,K))
		if log_space :
			betas[-1,:] = 0.
		else :
			betas[-1,:] = 1.

		#\alpha(z_1) = p(x_1,z_1) = p(x_1|z_1)p(z_1)
		for k in range(K) :
			if log_space :
				#alphas[0,k] = np.log(pi[k]) + np.log(emissions[0,k])
				alphas[0,k] = self.pi[k] + emissions[0,k]
			else :
				alphas[0,k] = self.pi[k] * emissions[0,k]
				
		for na,nb in zip(range(1,N),reversed(range(N-1))) :
			for k in range(K) :
				if log_space :
					alphas[na,k] = emissions[na,k]+\
					               self.lse(alphas[na-1,:]+self.A[:,k])
					betas[nb,k] = self.lse(emissions[nb+1,:]+\
						                   betas[nb+1,:]+\
						                   self.A[k,:])
					#alphas[na,k] = np.log(emissions[na,k])+lse(alphas[na-1,:]+np.log(A[:,k]))
					#betas[nb,k] = lse(np.log(emissions[nb+1,:])+betas[nb+1,:]+np.log(A[k,:]))
				else :
					alphas[na,k] = emissions[na,k]*\
					               np.sum(alphas[na-1,:]*self.A[:,k])
					betas[nb,k] = np.sum(emissions[nb+1,:]*\
						                 betas[nb+1,:]*\
						                 self.A[k,:])
		self.alphas = alphas.copy()
		### Calculate Responsibilities (Gammas) p(\bm{z}_i|\bm{X})###
		#\gamma(\bm{z}_i) = \frac{1}{p(\bm{X})}\alpha(\bm{z}_i)\beta(\bm{z}_i)
		#p(\bm{X}) = \sum_{\bm{z}_j\in Supp(\bm{z})}\alpha(\bm{z}_i)\beta(\bm{z}_i)
		#Note that p(\bm{X}) is also is a function of where you are on the chain (since it's effectively
		#normalizing the product \alpha(\bm{z}_i)\beta(\bm{z}_i)). So I find it easier to think of the following
		#p(\bm{X}) = Z(\bm{X},i)
		#\gamma(\bm{z}_i) = \frac{1}{Z(\bm{X},i)}\alpha(\bm{z}_i)\beta(\bm{z}_i)
		if log_space :
			gamma_no_norm = alphas + betas
		else :
			gamma_no_norm = alphas * betas
		# Here the normalization for the ith sample is
		#\frac{\gamma(\bm{z_}_i)}{sum_{\bm{z}_j}\gamma(\bm{z}_j)}
		#The "[:,np.newaxis]" is so that the N length normalization vector
		#is divided column-wise instead of row-wise (this is only a
		#consideration when N == K for some reason but comes up later)
		if log_space :
			gammas = gamma_no_norm-\
			         self.lse(gamma_no_norm,axis=1)[:,np.newaxis]
		else :
			gammas = gamma_no_norm/\
			         np.sum(gamma_no_norm,axis=1)[:,np.newaxis]

		### Calculate Xis ###
		#\xi(\bm{z}_{i-1},\bm{z}_i) = \frac{1}{p(\bm{X})}\alpha(\bm{z}_{i-1})
		#\beta(\bm{z}_i)p(\bm{x}_i|\bm{z}_i)p(\bm{z}_i|\bm{z}_{i-1})
		xi_no_norm = np.zeros((N-1,K,K))
		xis = xi_no_norm.copy()
		for n in range(1,N) :
			for k in range(K) :
				if log_space :
					xi_no_norm[n-1,k,:] = alphas[n-1,k]+\
					                      betas[n,:]+\
					                      emissions[n,:]+\
					                      self.A[k,:]
					#xi_no_norm[n-1,k,:] = alphas[n-1,k]+betas[n,:]+np.log(emissions[n,:])+np.log(A[k,:])
				else :
					xi_no_norm[n-1,k,:] = alphas[n-1,k]*\
					                      betas[n,:]*\
					                      emissions[n,:]*\
					                      self.A[k,:]
			if log_space :
				xis[n-1,:,:] =  xi_no_norm[n-1,:,:]-\
				                self.lse(xi_no_norm[n-1,:,:])
			else :
				xis[n-1,:,:] =  xi_no_norm[n-1,:,:]/\
				                np.sum(xi_no_norm[n-1,:,:])
		### Update Initial States (pi) ###
		if update_pi :
			if log_space :
				self.pi = gammas[0,:]-self.lse(gammas[0,:])
			else :
				self.pi = gammas[0,:]/np.sum(gammas[0,:])

		### Update Transition Matrix (A) ###
		if log_space :
			self.A = self.lse(xis,axis=0)-\
			         (self.lse(gammas[:-1,:],axis=0)[:,np.newaxis])
		else :
			self.A = np.sum(xis,axis=0)/\
			         np.sum(gammas[:-1,:],axis=0)[:,np.newaxis]

		self.update_emissions(**{'X':X,
			                     'gammas':gammas,
			                     'xis':xis,
			                     'log_space':log_space})

		if log_likelihood :
			if log_space :
				l = self.lse(alphas[-1,:])
			else :
				l = np.log(np.sum(alphas[-1,:]))

		if not return_normed :
			if log_likelihood :
				return gamma_no_norm,xi_no_norm,l
			else :
				return gamma_no_norm,xi_no_norm,None
		else :
			if log_likelihood :
				return gammas,xis,l
			else :
				return gammas,xis,None
				
	def emit(self,x,z,params={}) :
		pass

	def update_emissions(self,**kwargs) :
		pass

class CategoricalHMM(HMM) :
	def __init__(self,
		         A=None,
		         pi=None,
		         emissions=None) :
		super(CategoricalHMM,self).__init__(A=A,
			                                pi=pi,
			                                emissions=emissions)

	def emit(self,x,z,params={}) :
		return self.emissions[x,z]

	def update_emissions(self,**kwargs) :
		log_space = kwargs.get('log_space',False)
		X = kwargs.get('X',None)
		gammas = kwargs.get('gammas',None)
		if X is None or gammas is None :
			raise
		for i in range(self.emissions.shape[0]) :
			mask = np.squeeze(X) == i
			if log_space :
				self.emissions[i,:] = self.lse(gammas[mask,:],axis=0)-\
			                          self.lse(gammas,axis=0)
			else :
				self.emissions[i,:] = np.sum(gammas[mask,:],axis=0)/\
			                          np.sum(gammas,axis=0)

class GaussianHMM(HMM) :
	def __init__(self,
		         A=None,
		         pi=None,
		         emissions=None) :
		super(GaussianHMM,self).__init__(A=A,
			                             pi=pi,
			                             emissions=emissions)

	def emit(self,x,z,params={}) :
		"""
		Gaussian Emissions x_i|z_i ~ N(mu_{z_i},sigma_{z_i})
		self.emissions['means'].shape = (K,D)
		self.emissions['covs'].shape  = (K,D,D)
		where self.emissions['means'][k,:] = mu_k
		and self.emissions['covs'][k,:,:] = cov_k
		"""
		log_pdf = params.get('log_pdf','False')
		if log_pdf :
			return multivariate_normal.logpdf(x,
			                                  mean=self.emissions['means'][z,:],
			                                  cov=self.emissions['covs'][z,:,:])
		else :
			return multivariate_normal.pdf(x,
			                               mean=self.emissions['means'][z,:],
			                               cov=self.emissions['covs'][z,:,:])

	def update_emissions(self,**kwargs) :
		X = kwargs.get('X',None)
		gammas = kwargs.get('gammas',None)
		if X is None or gammas is None :
			raise
		K = gammas.shape[1]
		log_space = kwargs.get('log_space',False)
		eps = np.finfo(gammas.dtype).eps
		#(K,N)*(N,D)=(K,D)
		if log_space :
			denom = np.sum(np.exp(gammas),axis=0) + 10*eps
			self.emissions['means'] = np.exp(gammas).T.dot(X)/denom[:,np.newaxis]
		else :
			self.emissions['means'] = gammas.T.dot(X)/np.sum(gammas,axis=0)[:,np.newaxis]
		for k in range(K) :
			x_mu = X - self.emissions['means'][k,:] #(N,D) - (D,) = (N,D)
			if log_space :
				outer_prod = (np.exp(gammas[:,k])*x_mu.T).dot(x_mu)
				denom = np.sum(np.exp(gammas[:,k])) + 10*eps
				cov = outer_prod/denom
				cov[np.diag_indices_from(cov)] += 1e-6
				self.emissions['covs'][k,:,:] = cov
			else :
				outer_prod = (gammas[:,k]*x_mu.T).dot(x_mu)
				self.emissions['covs'][k,:,:] = outer_prod/np.sum(gammas[:,k])
			#self.emissions['covs'][k,:,:] = np.einsum('nd,nj->dj',gammas[:,k,np.newaxis]*x_mu,x_mu)/np.sum(gammas[:,k])

def main() :
	"""
	A = np.ones((2, 2),dtype=np.float64)
	A = A / np.sum(A, axis=1)
	A_l = np.log(A)
	b = np.array(((1, 3, 5), (2, 4, 6)),dtype=np.float64)
	b = b / np.sum(b, axis=1).reshape((-1, 1))
	b = b.T
	b_l = np.log(b)
	X = pd.read_csv('/home/chrism/sw/hmm/data/test.csv')
	X = X['Visible'].values[:,np.newaxis]

	c_hmm = CategoricalHMM(A=A,
		                   pi=np.array([0.5,0.5],dtype=np.float64),
		                   emissions=b)
	c_hmm.fit(X,
		      log_space=False,
		      stopping_criterion={'num_iters':100},
		      update_pi=False)
	c_hmm_l = CategoricalHMM(A=A_l,
		                     pi=np.log(np.array([0.5,0.5],dtype=np.float64)),
		                     emissions=b_l)
	c_hmm_l.fit(X,
		       log_space=True,
		       stopping_criterion={'num_iters':100},
		       update_pi=False)
	print(c_hmm.A)
	print(np.exp(c_hmm_l.A))
	"""
	A = np.ones((3,3))
	A /= 3
	A = np.log(A)
	b = {'means':np.array([[0,0],[5,5],[-5,5]],dtype=np.float64),
	     'covs':np.array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]],dtype=np.float64)}
	X1 = np.random.multivariate_normal(mean=b['means'][0,:],cov=b['covs'][0,:,:],size=300)
	X2 = np.random.multivariate_normal(mean=b['means'][1,:],cov=b['covs'][1,:,:],size=300)
	X3 = np.random.multivariate_normal(mean=b['means'][2,:],cov=b['covs'][2,:,:],size=300)
	X = np.concatenate((X1,X2,X3))
	#np.random.shuffle(X)
	b = {'means':np.array([[-1,1],[0.5,0.5],[-0.87,2.3]],dtype=np.float64),
	     'covs':np.array([[[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]],dtype=np.float64)}
	g_hmm_l = GaussianHMM(A=A,
		                  pi=np.log(np.array([0.5,0.5,0.5],dtype=np.float64)),
		                  emissions=b)
	g_hmm_l.fit(X,
		        log_space=True,
		        update_pi=False,
		        emission_params={'log_pdf':True})
	print(np.exp(g_hmm_l.A))
	print(g_hmm_l.emissions)
	print(np.exp(g_hmm_l.posterior_distribution(log_space=True)))
	#print(g_hmm_l.predictive_distribution(np.random.multivariate_normal(mean=b['means'][2,:],cov=b['covs'][2,:,:],size=1)))
	#"""

if __name__ == '__main__' :
	main()