import numpy as np
from numpy import fft
from scipy.linalg import dft
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import tensorflow as tf

def preprocess_complex_data(data):
    data_real = np.real(data)
    data_imag = np.imag(data)
    data_combined = np.stack((data_real, data_imag), axis=-1)
    return data_combined

def postprocess_complex_data(data):
    data = data.reshape(64,2)
    return data[:, 0] + 1j * data[:, 1]
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
def DFT_matrix(N):
    i, j = np.meshgrid(np.arange(N), np.arange(N))
    omega = np.exp( - 2 * np.pi * 1j / N )
    W = np.power( omega, i * j ) / np.sqrt(N)
    return W
def LS_estimator(OFDM_freq, pilotValue,pilotCarriers):
    
    H_LS = OFDM_freq[pilotCarriers]/pilotValue
    return H_LS

def MMSE_estimator(OFDM_freq, pilotValue,N,P,sigma2,covariance_matrix,pilotCarriers):
    
    X = pilotValue*np.eye(P,dtype='complex64')
    Y = OFDM_freq[pilotCarriers]
    
    H_LS = np.linalg.inv(X)@Y
    F = DFT_matrix(P)
  
    R_HH = F@((2*(covariance_matrix)**2)@(F.conj().T))

    H_MMSE = R_HH @ np.linalg.inv(R_HH + (sigma2)*np.linalg.inv(X @ X.conj().T)) @ H_LS

    return H_MMSE

    return H_est
def Demapping(QAM):
    # array of possible constellation points
    constellation = np.array([x for x in demapping_table.keys()])
    # print(constellation)
    # print(QAM)
    # calculate distance of each RX point to each possible point
    dists = abs(QAM.reshape((-1,1)) - constellation.reshape((1,-1)))
    
    # for each element in QAM, choose the index in constellation 
    # that belongs to the nearest constellation point
    const_index = dists.argmin(axis=1)
    
    # get back the real constellation point
    hardDecision = constellation[const_index]
    
    # transform the constellation point into the bit groups
    return np.vstack([demapping_table[C] for C in hardDecision]), hardDecision
### 16 QAM
def OFDM_Simulation(interpolation = 'Linear', estimator = 'LS',plot_H = False):
    M,N = 16,64
    m = int(np.log2(M))
    CP = 16 
    P = N//8
    ## 3 tap Rayleigh Channel
    L = 3
    pilotValue = 3+3j
    ## SNR Range from 0 to 30 dB
    SNR_db = np.arange(0,31,1)
    BER = np.zeros_like(SNR_db,dtype='float32')

    allCarriers = np.arange(N)  # indices of all subcarriers ([0, 1, ... K-1])

    pilotCarriers = allCarriers[::N//P] # Pilots is every (K/P)th carrier.
    
# For convenience of channel estimation, let's make the last carriers also be a pilot
    pilotCarriers = np.hstack([pilotCarriers, np.array([allCarriers[-1]])])
    P = P+1
    effective_N = N - P

# data carriers are all remaining carriers
    dataCarriers = np.delete(allCarriers, pilotCarriers)

    # print ("allCarriers:   %s" % allCarriers)
    # print ("pilotCarriers: %s" % pilotCarriers)
    # print ("dataCarriers:  %s" % dataCarriers)
    # plt.figure(figsize=(8,0.8))
    # plt.plot(pilotCarriers, np.zeros_like(pilotCarriers), 'bo', label='pilot')
    # plt.plot(dataCarriers, np.zeros_like(dataCarriers), 'ro', label='data')
    # plt.legend(fontsize=10, ncol=2)
    # plt.xlim((-1,K)); plt.ylim((-0.1, 0.3))
    # plt.xlabel('Carrier index')
    # plt.yticks([])
    # plt.grid(True);
    # model2 = tf.keras.models.load_model('model_weights_20_new.h5')
    if estimator == 'CNN':
        model1 = tf.keras.models.load_model('model_weights_20_new.h5')
        model2 = tf.keras.models.load_model('model_weights_30.h5')
        model3 = tf.keras.models.load_model('model_weights_10.h5')
        # print(model.summary())
    MSE = np.zeros_like(SNR_db,dtype='float32')
    #### Comb- Type
    for snr_db in SNR_db:
        blocks = 100
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
  
        
            cov = np.diag(avg_power)

            cov = np.vstack((np.hstack((cov,np.zeros((L,P-L)))),np.zeros((P-L,P))))

            channelResponse = avg_power*np.random.randn(1)+1j*avg_power*np.random.randn(1)
            # h0 = np.random.normal(0,0.3/2,1)+1j*np.random.normal(0,0.3/2,1)
            # h1 = np.random.normal(0,0.8/2,1)+1j*np.random.normal(0,0.8/2,1)
            # h2 = np.random.normal(0,0.2/2,1)+1j*np.random.normal(0,0.2/2,1)
            # # h0 = np.random.randn(1)+1j*np.random.randn(1)
            # # h1 = np.random.randn(1)+1j*np.random.randn(1)
            # # h2 = np.random.randn(1)+1j*np.random.randn(1)
            # channelResponse = np.array([h0,h1,h2])  # the impulse response of the wireless channel
            channelResponse = np.array([1, 0, 0.3+0.3j])
            channelResponse = channelResponse.reshape(1, -1)
            H_exact = np.fft.fft(channelResponse, N).reshape(-1,1)
    

            output = np.convolve(OFDM_time_CP.ravel(),channelResponse.ravel())[:N+CP]
            signal_power = np.mean(abs(output**2))
            sigma2 = signal_power * 10**(-snr_db/10)  # calculate noise power based on signal power and SNR
        
            # print ("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))
        
        # Generate complex noise with given variance
            noise = np.sqrt(sigma2/2) * (np.random.randn(*output.shape)+1j*np.random.randn(*output.shape))

            OFDM_RX = output + noise
            # outpu


            plot_TX_RX = False
            if plot_TX_RX :
                
                plt.figure(figsize=(8,2))
                plt.plot(abs(OFDM_time_CP), label='TX signal')
                plt.plot(abs(OFDM_RX), label='RX signal')
                plt.legend(fontsize=10)
                plt.xlabel('Time')
                plt.ylabel('$|x(t)|$')
                plt.show()


            OFDM_RX_noCP = OFDM_RX[CP:(CP+N)]
            OFDM_freq  = np.fft.fft(OFDM_RX_noCP)
            
            if estimator == 'LS':
                H_est_pilots = LS_estimator(OFDM_freq, pilotValue, pilotCarriers)
            elif estimator == 'MMSE':
                H_est_pilots = MMSE_estimator(OFDM_freq, pilotValue, N, P,sigma2,cov, pilotCarriers)
                        
            if estimator == 'exact':
                Hest = H_exact
            elif estimator == 'CNN':

                data = preprocess_complex_data(OFDM_freq)

                data = data.reshape(1,64,2)
                # if snr_db <=25 :
                #     estimate = model1.predict(data,verbose = 0)
                # else:
                estimate = model2.predict(data,verbose = 0)
                Hest = postprocess_complex_data(estimate)
              
            elif interpolation == 'Linear':
                ## Channel Estimation - Linear Interpolation
                
                Hest_abs = interp1d(pilotCarriers, abs(H_est_pilots), kind='linear')(allCarriers)
                Hest_phase = interp1d(pilotCarriers, np.angle(H_est_pilots), kind='linear')(allCarriers)
                Hest = Hest_abs * np.exp(1j*Hest_phase)
  
            elif interpolation == 'Quadratic':
                ## Channel Estimation - Quadratic Interpolation
                Hest_abs = interp1d(pilotCarriers, abs(H_est_pilots), kind='quadratic')(allCarriers)
                Hest_phase = interp1d(pilotCarriers, np.angle(H_est_pilots), kind='quadratic')(allCarriers)
                Hest = Hest_abs * np.exp(1j*Hest_phase)
            
            elif interpolation == 'Cubic':
                ## Channel Estimation - Cubic Interpolation
                Hest_abs = interp1d(pilotCarriers, abs(H_est_pilots), kind = 'cubic')(allCarriers)
                Hest_phase = interp1d(pilotCarriers, abs(H_est_pilots), kind = 'cubic')(allCarriers)
                Hest = Hest_abs * np.exp(1j*Hest_phase)
            H_exact_pilot = H_exact[pilotCarriers]

            
            if plot_H:
                plt.suptitle("SNR = %.1f dB" % snr_db)
                plt.plot(allCarriers, abs(H_exact), label='True channel')
                plt.plot(allCarriers, abs(Hest), label='Least Squares')
     
                H_est_pilots = MMSE_estimator(OFDM_freq, pilotValue, N, P,sigma2,cov,pilotCarriers)
                Hest_abs = interp1d(pilotCarriers, abs(H_est_pilots), kind='quadratic')(allCarriers)
                Hest_phase = interp1d(pilotCarriers, np.angle(H_est_pilots), kind='quadratic')(allCarriers)
                Hest = Hest_abs * np.exp(1j*Hest_phase)

                plt.plot(allCarriers, abs(Hest), label='MMSE ')
                data = preprocess_complex_data(OFDM_freq)
                    # print(data.shape)
                data = data.reshape(1,64,2)
                # if snr_db <=25 :
                #     estimate = model1.predict(data,verbose = 0)
                # else:
                estimate = model2.predict(data,verbose = 0)
                Hest = postprocess_complex_data(estimate)
                plt.plot(allCarriers, abs(Hest), label='CNN Based')
                plt.xlabel('Subcarrier Index')
                plt.ylabel('$|H|$')
                plt.legend()
 
            MSE[index] +=  np.mean(np.abs((H_exact-Hest))**2)
            equalised = OFDM_freq/Hest.ravel()


            data_RX = equalised[dataCarriers]
            PS_est, hardDecision = Demapping(data_RX.ravel())

            RX_bits = PS_est.reshape(1,-1)

            berror = np.sum(np.abs(RX_bits - info_bits))
            # print('block error = ', berror)
            error += berror
        BER[index] = error/(blocks*(effective_N*m))
        MSE[index] = MSE[index]/(blocks*N)

    # plt.semilogy(SNR_db,BER)
    # plt.xlabel('SNR')
    # plt.ylabel('BER')
    # if estimator != 'CNN':
    #        plt.plot(SNR_db,BER,label = str(estimator) + ' Estimator with ' + str(interpolation) + ' Interpolation')
    # else :
    #     plt.plot(SNR_db,BER,label = str(estimator) + ' Estimator')

    plt.xlabel('SNR')
    plt.ylabel('MSE')
    if estimator != 'CNN':
        plt.plot(SNR_db,MSE,label = str(estimator) + ' Estimator with ' + str(interpolation) + ' Interpolation')
    else :
        plt.plot(SNR_db,MSE,label = str(estimator) + ' Estimator')

    plt.legend()

import time
start_time = time.time()
OFDM_Simulation(interpolation= 'Quadratic',estimator= 'LS')
# time1  = time.time()
# print(time1 - start_time)
OFDM_Simulation(interpolation= 'Quadratic',estimator= 'MMSE')
# time2 = time.time()
# print(time2-time1)
# # OFDM_Simulation(interpolation= 'Linear',estimator= 'MMSE')
# time3 = time.time()
# print(time3-time2)
# # OFDM_Simulation(interpolation= 'Linear',estimator= 'LS')
# time4 = time.time()
# print(time4-time3)
# OFDM_Simulation(estimator= 'exact')
OFDM_Simulation(estimator= 'CNN')
# time5 = time.time()
# print(time5-time4)
# OFDM_Simulation(interpolation= 'Cubic',estimator='MMSE')
# OFDM_Simulation(interpolation= 'Cubic',estimator='LS')
plt.show()