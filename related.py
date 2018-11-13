import math
from scipy import stats as stat
import numpy as np
import random



def frequencies(nsites):
    """simulate allele frequencies"""    
    frequencies = []
    for i in range(nsites):
        frequencies.append(random.uniform(0.05, 0.95))
    return frequencies

def haplotypes(frequencies, ndip):
    """simulating haplotypes based on the previous simulated allele frequencies"""
    haplo = []
    for freq in frequencies:
        haparr = np.random.choice(np.array([1,0]),ndip*2,p=[freq,1-freq])
        haplo.append(haparr.tolist())
    return haplo


# test genotype, genotype likelihood is the probability that a certain genotype is correct given the data



def getlikes(idv, depth, e=0.01, norm=False):
    """function to compute genotype likelihoods"""
    geno = np.sum(idv,axis=1)
    n = geno.shape[0]
    dep = np.random.poisson(depth,n)
    #number of successes in a given amount of trials, in our case the number of successes needs to be every site (nsites), effectively giving the probability of observing a certain genotype.
    nA = np.random.binomial(dep,np.array([e,0.5,1-e])[geno],n)
    res = np.transpose(np.array([stat.binom.pmf(nA,dep,e),stat.binom.pmf(nA,dep,0.5),stat.binom.pmf(nA,dep,1-e)]))
    if norm == True:
        np.divide(res,np.sum(res, axis=1).reshape(res.shape[0],1))
    return res



def perfectlikes(genovec):
    genomatrix = np.zeros((genovec.shape[0],3))
    genomatrix[np.where(np.sum(genovec, axis=1)==0),0]=1
    genomatrix[np.where(np.sum(genovec, axis=1)==1),1]=1
    genomatrix[np.where(np.sum(genovec, axis=1)==2),2]=1
    
    return genomatrix



# checking that population frequencies and sample frequencies are reasonably similar


def freq_val(frequency, haplotype):

    frequency_validation = []
    for i, j in zip(frequency, haplotype):
        frequency_validation.append(abs(np.mean(j)-i))
    return frequency_validation
    


def makegamet(haplotypes, individual_start_index, individual_stop_index):
    "takes two haplotypes from one individual as input and outputs a gamet"
    gamet = []
    for site in haplotypes:
        gamet.append(random.choice(site[individual_start_index:individual_stop_index]))
    return gamet




def emis9(idv1, idv2, freq_arr):    #freq is the frequency of 1=one (1=i in article) not 0=zero (0=j in article)
    """calculating the per-site probability of the nine different IBD modes (jacquard coefficients) based on the IBS modes"""
    g1 = np.sum(idv1, axis=1)
    g2 = np.sum(idv2, axis=1)
    freq1 = freq_arr
    freq0 = 1-freq_arr
    zero_arr = np.zeros(freq_arr.shape[0])
    
    emis = np.zeros((freq_arr.shape[0],9))
    
    
    #00&00 IBS mode S1
    keep = g1+g2==0  #pairwise check of the two arrays to see if the sum is 0, gives an array of boolean values true or false
    tmp0 = freq0[keep]
    emis[keep] = np.transpose(np.array([tmp0,tmp0**2,tmp0**2,tmp0**3,tmp0**2,tmp0**3,tmp0**2,tmp0**3,tmp0**4]))
    
    #11&11 IBS mode S1
    keep = g1+g2==4
    tmp1 = freq1[keep]
    emis[keep] = np.transpose(np.array([tmp1,tmp1**2,tmp1**2,tmp1**3,tmp1**2,tmp1**3,tmp1**2,tmp1**3,tmp1**4]))
    
    #11&00 IBS mode s2
    keep = np.logical_and(g1==2,g2==0)
    tmp0 = freq0[keep]
    tmp1 = freq1[keep]
    zeros = zero_arr[keep]
    emis[keep] = np.transpose(np.array([zeros,tmp1*tmp0,zeros,tmp1*tmp0**2,zeros,tmp1**2*tmp0,zeros,zeros,tmp1**2*tmp0**2]))
    
    #00&11 IBS mode S2
    keep = np.logical_and(g1==0,g2==2)
    tmp0 = freq0[keep]
    tmp1 = freq1[keep]
    zeros = zero_arr[keep]
    emis[keep] = np.transpose(np.array([zeros,tmp0*tmp1,zeros,tmp0*tmp1**2,zeros,tmp0**2*tmp1,zeros,zeros,tmp0**2*tmp1**2]))
    
    #11&01 IBS mode S3
    keep = np.logical_and(g1==2,g2==1)
    tmp0 = freq0[keep]
    tmp1 = freq1[keep]
    zeros = zero_arr[keep]
    emis[keep] = np.transpose(np.array([zeros,zeros,tmp1*tmp0,2*tmp1**2*tmp0,zeros,zeros,zeros,tmp1**2*tmp0,2*tmp1**3*tmp0]))
    
    #00&01 IBS mode S3
    keep = np.logical_and(g1==0,g2==1)
    tmp0 = freq0[keep]
    tmp1 = freq1[keep]
    zeros = zero_arr[keep]
    emis[keep] = np.transpose(np.array([zeros,zeros,tmp0*tmp1,2*tmp0**2*tmp1,zeros,zeros,zeros,tmp0**2*tmp1,2*tmp0**3*tmp1]))
    
    #01&11 IBS mode S5
    keep = np.logical_and(g1==1,g2==2)
    tmp0 = freq0[keep]
    tmp1 = freq1[keep]
    zeros = zero_arr[keep]
    emis[keep] = np.transpose(np.array([zeros,zeros,zeros,zeros,tmp1*tmp0,2*tmp1**2*tmp0,zeros,tmp1**2*tmp0,2*tmp1**3*tmp0]))
    
    #01&00 IBS mode S5
    keep = np.logical_and(g1==1,g2==0)
    tmp0 = freq0[keep]
    tmp1 = freq1[keep]
    zeros = zero_arr[keep]
    emis[keep] = np.transpose(np.array([zeros,zeros,zeros,zeros,tmp0*tmp1,2*tmp0**2*tmp1,zeros,tmp0**2*tmp1,2*tmp0**3*tmp1]))

    #01&01 IBS mode S7
    keep = np.logical_and(g1==1,g2==1)
    tmp0 = freq0[keep]
    tmp1 = freq1[keep]
    zeros = zero_arr[keep]
    emis[keep] = np.transpose(np.array([zeros,zeros,zeros,zeros,zeros,zeros,2*tmp0*tmp1,np.multiply(tmp0*tmp1,tmp1+tmp0),4*tmp1**2*tmp0**2]))
    
    return emis
    


def emis9_gl(gl1, gl2, freq_arr):
    
    """calculating the per-site probability of the nine different IBD modes (jacquard coefficients) based on the IBS modes and genotype likelihoods"""
   
    #a global likelihood can then be found later by taking the product of the persite likelihoods in the emission matrix as each site is independent of the others
    
    #gl1[:,0] er genotype likelihoods for at et site er af genotypen 00 i individ 1
    #gl2[:,0] er genotype likelihoods for at et site er af genotypen 00 i individ 2
    #gl1[:,1] er genotype likelihoods for at et site er af genotypen 01 i individ 1
    #gl2[:,1] er genotype likelihoods for at et site er af genotypen 01 i individ 2
    #gl1[:,2] er genotype likelihoods for at et site er af genotypen 11 i individ 1
    #gl2[:,2] er genotype likelihoods for at et site er af genotypen 11 i individ 2
    
    freq1 = freq_arr
    
    freq0 = 1-freq_arr
    
    zeros = np.zeros(gl1.shape[0])
    
    
    #00&00 IBS mode S1
    emis = np.transpose(np.array([freq0,freq0**2,freq0**2,freq0**3,freq0**2,freq0**3,freq0**2,freq0**3,freq0**4]))*gl1[:,0].reshape(gl1.shape[0],1)*gl2[:,0].reshape(gl2.shape[0],1)
    
    #11&11 IBS mode S1
    emis += np.transpose(np.array([freq1,freq1**2,freq1**2,freq1**3,freq1**2,freq1**3,freq1**2,freq1**3,freq1**4]))*gl1[:,2].reshape(gl1.shape[0],1)*gl2[:,2].reshape(gl2.shape[0],1)
    
    #11&00 IBS mode s2
    emis += np.transpose(np.array([zeros,freq1*freq0,zeros,freq1*freq0**2,zeros,freq1**2*freq0,zeros,zeros,freq1**2*freq0**2]))*gl1[:,2].reshape(gl1.shape[0],1)*gl2[:,0].reshape(gl2.shape[0],1)
    
    #00&11 IBS mode S2
    emis += np.transpose(np.array([zeros,freq0*freq1,zeros,freq0*freq1**2,zeros,freq0**2*freq1,zeros,zeros,freq0**2*freq1**2]))*gl1[:,0].reshape(gl1.shape[0],1)*gl2[:,2].reshape(gl2.shape[0],1)
    
    #11&01 IBS mode S3
    emis += np.transpose(np.array([zeros,zeros,freq1*freq0,2*freq1**2*freq0,zeros,zeros,zeros,freq1**2*freq0,2*freq1**3*freq0]))*gl1[:,2].reshape(gl1.shape[0],1)*gl2[:,1].reshape(gl2.shape[0],1)
    
    #00&01 IBS mode S3
    emis += np.transpose(np.array([zeros,zeros,freq0*freq1,2*freq0**2*freq1,zeros,zeros,zeros,freq0**2*freq1,2*freq0**3*freq1]))*gl1[:,0].reshape(gl1.shape[0],1)*gl2[:,1].reshape(gl2.shape[0],1)
    
    #01&11 IBS mode S5
    emis += np.transpose(np.array([zeros,zeros,zeros,zeros,freq1*freq0,2*freq1**2*freq0,zeros,freq1**2*freq0,2*freq1**3*freq0]))*gl1[:,1].reshape(gl1.shape[0],1)*gl2[:,2].reshape(gl2.shape[0],1)
    
    #01&00 IBS mode S5
    emis += np.transpose(np.array([zeros,zeros,zeros,zeros,freq0*freq1,2*freq0**2*freq1,zeros,freq0**2*freq1,2*freq0**3*freq1]))*gl1[:,1].reshape(gl1.shape[0],1)*gl2[:,0].reshape(gl2.shape[0],1)

    #01&01 IBS mode S7
    emis += np.transpose(np.array([zeros,zeros,zeros,zeros,zeros,zeros,2*freq0*freq1,np.multiply(freq0*freq1,freq1+freq0),4*freq1**2*freq0**2]))*gl1[:,1].reshape(gl1.shape[0],1)*gl2[:,1].reshape(gl2.shape[0],1)
    
    return emis




def llh(theta, emis):
    """the log likelihood function we are trying to find a lower bound for in the E step of the EM-algorithm, theta is the parameters and emis is the observed data, they are both numpy arrays"""    
    if theta.shape[0] == emis.shape[1]:
        llh = -np.sum(np.log(np.sum(theta*emis, axis=1)))
        return llh
    else:
        print ("cannot multiply arrays element-wise as shape of arrays has to be compatible")





def mstep(theta, emis):
    """the maximization step of the EM-algorithm"""
    temp = theta*emis #element-wise multiplication of the two numpy arrays
    temp = temp.T/np.sum(temp, axis=1)
    next_theta = np.mean(temp, axis=1)
    return next_theta



def em(theta, emis, niter=20000, tol=0.0000001):

    """expectation maximization algorithm"""    
    
    lasttheta = theta
    lastllh = llh(lasttheta, emis)
    
    for i in range(niter):
        newtheta = mstep(lasttheta, emis)
        print ("newtheta is:",newtheta)
        newllh = llh(newtheta, emis)
        print ("newllh is:",newllh)
        if (abs(lastllh-newllh)<tol):
            print ("breaking at iteration", i+1)
            lasttheta = newtheta
            lastllh = newllh
            break
        lasttheta = newtheta
        lastllh = newllh
    
    return lasttheta
        


def emaccel(theta, emis, niter=1500, tol=0.00001):
    "accelerated expectation maximization algorithm"
    
    min_step = 1
    max0_step = 1
    max_step = 1
    emstep = 4
    iteration = 1 
    p = theta
    llh_old = llh(p, emis)
    llh_eval = 1
    feval = 0
    
    
    while feval < niter:
        
        p1 = mstep(p, emis)
        feval = feval + 1
        q1 = p1-p
        sr2 = np.dot(q1,q1)
        if np.sqrt(sr2) < tol:
            p=p1
            break
        
        p2 = mstep(p1, emis)
        feval = feval + 1
        q2 = p2 - p1
        sq2 = np.sqrt(np.dot(q2,q2))
        if sq2 < tol:
            p=p2
            break
        
        sv2 = np.dot(q2-q1, q2-q1)
        srv = np.dot(q1, q2-q1)
        alpha = np.sqrt(np.divide(sr2,sv2))
        alpha = max(min_step, min(max_step,alpha))
        p_new = p + 2 * alpha * q1 + alpha**2 * (q2-q1)
        #print ("p_new is:",p_new)
        
        if np.any(p_new<0) or np.any(p_new>1):
            #print ("problem, p_new is out of parameterspace")
            p_new = p2
            
        if abs(alpha-1) > 0.01:
            p_new = mstep(p_new, emis)
            feval = feval+1
            
        llh_new = llh(p_new, emis)
        #print ("llh_new is:",llh_new)
        llh_eval = llh_eval+1
        
        if alpha == max_step:
            max_step = emstep * max_step
        
        if min_step < 0 and alpha == min_step:
            min_step = emstep * min_step
        p = p_new
        
        if not math.isnan(llh_new):
            llh_old = llh_new
            #print ("llh_old is:",llh_old)
        iteration = iteration + 1
        #print ("iteration is:",iteration)
    
    llh_old = llh(p, emis)
    llh_eval = llh_eval + 1
    
    return p,iteration



# function for plotting boxplots of the nine jacquard coefficients for identical twins
def ident_twin(theta, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []

    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = perfectlikes(b1)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl1,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append[jacq_coef[1]]

    ident_twins_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return ident_twins_arrays, iterations

# function for plotting boxplots of the nine jacquard coefficients for siblings
def sibling(theta, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        b2 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = perfectlikes(b1)
        gl2 = perfectlikes(b2)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    sibling_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return sibling_arrays,iterations




# function for plotting boxplots of the nine jacquard coefficients for father-offspring
def father_off(theta, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = perfectlikes(b1)
        gl2 = perfectlikes(f)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    father_off_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return father_off_arrays,iterations



# function for plotting boxplots of the nine jacquard coefficients for mother-offspring
def mother_off(theta, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = perfectlikes(b1)
        gl2 = perfectlikes(m)


        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    mother_off_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return mother_off_arrays,iterations




# function for plotting boxplots of the nine jacquard coefficients for half siblings

def half_sib(theta, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        hs = np.transpose(np.array([makegamet(m, 0, 2),makegamet(haptyp, 16, 18)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = perfectlikes(b1)
        gl2 = perfectlikes(hs)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    half_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return half_arrays,iterations




# function for plotting boxplots of the nine jacquard coefficients for cousins

def cousin(theta, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        m2 = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        c1 = np.transpose(np.array([makegamet(m2, 0, 2),makegamet(haptyp, 18, 20)]))
        
        gl1 = perfectlikes(b1)
        gl2 = perfectlikes(c1)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    cousin_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return cousin_arrays,iterations




# function for plotting boxplots of the nine jacquard coefficients for identical twins
def ident_twin_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = getlikes(b1,depth,error)

        jacq_coef = em(theta,emis9_gl(gl1,gl1,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    ident_twin_gl_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return ident_twin_gl_arrays,iterations




# function for plotting boxplots of the nine jacquard coefficients for siblings
def sibling_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        b2 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = getlikes(b1,depth,error)
        gl2 = getlikes(b2,depth,error)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    sibling_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return sibling_arrays,iterations





# function for plotting boxplots of the nine jacquard coefficients for father-offspring
def father_off_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = getlikes(b1,depth,error)
        gl2 = getlikes(f,depth,error)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    father_off_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return father_off_arrays,iterations



# function for plotting boxplots of the nine jacquard coefficients for mother-offspring
def mother_off_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = getlikes(b1,depth,error)
        gl2 = getlikes(m,depth,error)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    mother_off_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return mother_off_arrays,iterations




# function for plotting boxplots of the nine jacquard coefficients for half siblings

def half_sib_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        hs = np.transpose(np.array([makegamet(m, 0, 2),makegamet(haptyp, 16, 18)]))
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = getlikes(b1,depth,error)
        gl2 = getlikes(hs,depth,error)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef)
        
    half_arrays = np.array(array_list)
    
    return half_arrays




# function for plotting boxplots of the nine jacquard coefficients for cousins

def cousin_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):
    """takes all the neccessary parameters for the previous functions to predict the nine jacquard coefficients and executes nruns times to make a boxplot of the coefficients"""
    
    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)
        
        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        m2 = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        c1 = np.transpose(np.array([makegamet(m2, 0, 2),makegamet(haptyp, 18, 20)]))
        
        gl1 = getlikes(b1,depth,error)
        gl2 = getlikes(c1,depth,error)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    cousin_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return cousin_arrays,iterations



def selfsib(theta, nruns=100, nsites=100000, ndip=20):

    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)

        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        sb1 = np.transpose(np.array([makegamet(m, 0, 2),makegamet(m, 0, 2)]))
        sb2 = np.transpose(np.array([makegamet(m, 0, 2),makegamet(m, 0, 2)]))
        
        gl1 = perfectlikes(sb1)
        gl2 = perfectlikes(sb2)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    selfsib_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return selfsib_arrays,iterations


def selfsib_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):

    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)

        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))

        selfb1 = np.transpose(np.array([makegamet(m, 0, 2),makegamet(m, 0, 2)]))
        selfb2 = np.transpose(np.array([makegamet(m, 0, 2),makegamet(m, 0, 2)]))

        gl1 = getlikes(selfb1,depth,error)
        gl2 = getlikes(selfb2,depth,error)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    selfsib_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return selfsib_arrays,iterations


def sibsib(theta, nruns=100, nsites=100000, ndip=20):

    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)

        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        b2 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        sb1 = np.transpose(np.array([makegamet(b1, 0, 2),makegamet(b2, 0, 2)]))
        sb2 = np.transpose(np.array([makegamet(b1, 0, 2),makegamet(b2, 0, 2)]))

        gl1 = perfectlikes(sb1)
        gl2 = perfectlikes(sb2)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    sibsib_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return sibsib_arrays,iterations


def sibsib_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):

    array_list = []
    iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)

        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        b2 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        sb1 = np.transpose(np.array([makegamet(b1, 0, 2),makegamet(b2, 0, 2)]))
        sb2 = np.transpose(np.array([makegamet(b1, 0, 2),makegamet(b2, 0, 2)]))

        gl1 = getlikes(sb1,depth,error)
        gl2 = getlikes(sb2,depth,error)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    sibsib_gl_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return sibsib_gl_arrays,iterations


def sibsibsib(theta, nruns=100, nsites=100000, ndip=20):

    array_list = []
	iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)

        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        b2 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        sb1 = np.transpose(np.array([makegamet(b1, 0, 2),makegamet(b2, 0, 2)]))
        sb2 = np.transpose(np.array([makegamet(b1, 0, 2),makegamet(b2, 0, 2)]))
        ssb1 = np.transpose(np.array([makegamet(sb1, 0, 2),makegamet(sb2, 0, 2)]))
        ssb2 = np.transpose(np.array([makegamet(sb1, 0, 2),makegamet(sb2, 0, 2)]))
        
        gl1 = perfectlikes(ssb1)
        gl2 = perfectlikes(ssb2)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    sibsibsib_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return sibsibsib_arrays,iterations

def sibsibsib_gl(theta, depth, error, nruns=100, nsites=100000, ndip=20):

    array_list = []
	iteration_list = []
    for i in range(nruns):
        
        
        freq = frequencies(nsites)
        freq_arr = np.array(freq)
        haptyp = haplotypes(freq,ndip)

        fm = np.transpose(np.array([makegamet(haptyp, 8, 10),makegamet(haptyp, 10, 12)]))
        ff = np.transpose(np.array([makegamet(haptyp, 12, 14),makegamet(haptyp, 14, 16)]))
        mf = np.transpose(np.array([makegamet(haptyp, 4, 6),makegamet(haptyp, 6, 8)]))
        mm = np.transpose(np.array([makegamet(haptyp, 0, 2),makegamet(haptyp, 2, 4)]))
        m = np.transpose(np.array([makegamet(mm, 0, 2),makegamet(mf, 0, 2)]))
        f = np.transpose(np.array([makegamet(ff, 0, 2),makegamet(fm, 0, 2)]))
        b1 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        b2 = np.transpose(np.array([makegamet(f, 0, 2),makegamet(m, 0, 2)]))
        sb1 = np.transpose(np.array([makegamet(b1, 0, 2),makegamet(b2, 0, 2)]))
        sb2 = np.transpose(np.array([makegamet(b1, 0, 2),makegamet(b2, 0, 2)]))
        ssb1 = np.transpose(np.array([makegamet(sb1, 0, 2),makegamet(sb2, 0, 2)]))
        ssb2 = np.transpose(np.array([makegamet(sb1, 0, 2),makegamet(sb2, 0, 2)]))
        
        gl1 = getlikes(ssb1,depth,error)
        gl2 = getlikes(ssb2,depth,error)

        jacq_coef = emaccel(theta,emis9_gl(gl1,gl2,freq_arr))
        
        array_list.append(jacq_coef[0])
        iteration_list.append(jacq_coef[1])
    sibsibsib_gl_arrays = np.array(array_list)
    iterations = np.sum(np.array(iteration_list))/100
    return sibsibsib_gl_arrays,iterations