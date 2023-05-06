import numpy as np
from numpy import fft
from scipy.linalg import dft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf

## 16 QAM Mapping Table
mapping_table = {
    (0,0,0,0) : -3-3j,
    (0,0,0,1) : -3-1j,
    (0,0,1,0) : -3+3j,
    (0,0,1,1) : -3+1j,
    (0,1,0,0) : -1-3j,
    (0,1,0,1) : -1-1j,
    (0,1,1,0) : -1+3j,
    (0,1,1,1) : -1+1j,
    (1,0,0,0) :  3-3j,
    (1,0,0,1) :  3-1j,
    (1,0,1,0) :  3+3j,
    (1,0,1,1) :  3+1j,
    (1,1,0,0) :  1-3j,
    (1,1,0,1) :  1-1j,
    (1,1,1,0) :  1+3j,
    (1,1,1,1) :  1+1j
}

demapping_table = {v : k for k, v in mapping_table.items()}

def LS_estimator(OFDM_freq, pilotValue,pilotCarriers):
    
    H_LS = OFDM_freq[pilotCarriers]/pilotValue
    return H_LS

    return H_est
def Demapping(QAM):
    
    constellation = np.array([x for x in demapping_table.keys()])
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))

    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision
### 16 QAM
def OFDM_Simulation(interpolation = 'Linear', estimator = 'LS'):
    M,N = 16,64
    m = int(np.log2(M))
    CP = 16 
    P = N//8
    ## 3 tap Rayleigh Channel
    L = 3
    pilotValue = 3+3j
    ## SNR Range from 0 to 30 dB
    SNR_db = np.arange(15,16,1)
    BER = np.zeros_like(SNR_db,dtype='float32')

    allCarriers = np.arange(N)  # indices of all subcarriers ([0, 1, ... K-1])

    pilotCarriers = allCarriers[::N//P] # Pilots is every (K/P)th carrier.
    
# For convenience of channel estimation, let's make the last carriers also be a pilot
    pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
    P = P+1
    effective_N = N - P

# data carriers are all remaining carriers
    dataCarriers = np.delete(allCarriers, pilotCarriers)

    #### Comb- Type
    for snr_db in SNR_db:
        blocks = 1
        error = 0
        index = snr_db - SNR_db[0]

        for t in range(blocks):
            
            ## Information Bits
            info_bits = np.random.randint(2,size = (1,effective_N*m))
            info_bits_blocks = info_bits.reshape((-1, m))
            data = np.zeros((effective_N,1),dtype='complex64')
            for i in range(effective_N):
                data[i] = mapping_table[tuple(info_bits_blocks[i])]        

            OFDM_data = np.zeros((N,1),dtype='complex64')
 

            OFDM_data[pilotCarriers] = pilotValue
            OFDM_data[dataCarriers] = data




            OFDM_time = np.fft.ifft(OFDM_data.ravel()).reshape(-1,1)
    
            
            cp = OFDM_time[-CP:]   # take the last CP samples ...

            OFDM_time_CP = np.vstack([cp, OFDM_time])


            ### Channel 
            ## 3 Tap Channel with Fixed Avg Power
            avg_power = np.array([0.3,0.8,0.2])
            # avg_power = np.array([1,1,1])


            channelResponse = avg_power*np.random.randn(1)+1j*avg_power*np.random.randn(1)

            channelResponse = channelResponse.reshape(1, -1)
            H_exact = np.fft.fft(channelResponse, N).reshape(-1,1)
    

            output = np.convolve(OFDM_time_CP.ravel(),channelResponse.ravel())[:N+CP]
            signal_power = np.mean(abs(output**2))
            sigma2 = signal_power * 10**(-snr_db/10)  # calculate noise power based on signal power and SNR
        
            # print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
        
        # Generate complex noise with given variance
            noise = np.sqrt(sigma2/2) * (np.random.randn(*output.shape)+1j*np.random.randn(*output.shape))

            OFDM_RX = output + noise
            # OFDM_RX = output

            OFDM_RX_noCP = OFDM_RX[CP:(CP+N)]
            OFDM_freq  = np.fft.fft(OFDM_RX_noCP)
                        
            H_est_pilots = LS_estimator(OFDM_freq, pilotValue, pilotCarriers)
            

        
            
            ## Channel Estimation - Quadratic Interpolation
            Hest_abs = interp1d(pilotCarriers, abs(H_est_pilots), kind='quadratic')(allCarriers)
            Hest_phase = interp1d(pilotCarriers, np.angle(H_est_pilots), kind='quadratic')(allCarriers)
            Hest = Hest_abs * np.exp(1j*Hest_phase)

   
            Ht = np.real(np.fft.ifft(Hest))
            sigma_est = np.sqrt(2/np.log(4)) * np.median(Ht)
            T = np.sqrt(2*np.log(16)) * sigma_est
     
            c = Ht[Ht<=T]
            L_ = len(c)
            sigma_est2 = np.sqrt(2/np.log(4)) *np.median(c)
            T2 = np.sqrt(2*np.log(L_)) *sigma_est2    
            Ht[Ht <= T2] = 0
            Hest_sigma = np.fft.fft(Ht, 64)
            # print(H_exact_pilot)
            # print(np.mean(H_exact_pilot-H_est_pilots))
                
            plt.suptitle("SNR = %.1f dB" % snr_db)
            plt.plot(allCarriers, abs(H_exact), label='True channel')
            plt.plot(allCarriers, abs(Hest), label='Least Squares')
            plt.plot(allCarriers, abs(Hest_sigma), label='Least Squares with Channel Tap Estimate')
            # plt.plot(allCarriers, abs(Hf), label='Least Squares with Channel Tap Estimate (first stage)')
            plt.legend()
            # plt.show()
           
            equalised = OFDM_freq/Hest.ravel()


            data_RX = equalised[dataCarriers]
            

            PS_est, hardDecision = Demapping(data_RX.ravel())

         

            RX_bits = PS_est.reshape(1,-1)

            berror = np.sum(np.abs(RX_bits - info_bits))
            error += berror



import time
start_time = time.time()
OFDM_Simulation(interpolation= 'Quadratic',estimator= 'LS')

plt.show()