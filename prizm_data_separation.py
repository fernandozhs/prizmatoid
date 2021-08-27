from matplotlib import pyplot as plt
import numpy as np
import prizmatoid as pzt

if __name__ == '__main__':

    # Initial and final ctimes associated with the data of interest.
    initial_ctime = 1629400000
    final_ctime = 1629500000
    
    # Loads the data into a Python dictionary.
    prizm_data = pzt.read_prizm_data(initial_ctime, final_ctime,
                                     dir_top='/Users/Fernando/Documents/Data/PRIZM/marion2021',
                                     subdir_100='data_100MHz', subdir_70='data_70MHz', subdir_switch='switch_data',
                                     read_100=True, read_70=False, read_switch=True, read_temp=False,
                                     verbose=False)
    
    # Add switch flags to the dictionary. These will be used below to separate the different data components.
    pzt.add_switch_flags(prizm_data, antennas=['100MHz'])
    
    # Creates data-selecting filters from the switch flags. Here we use a buffer `(a, b)` to eliminate boundary `a` data points at the  beginning and `b` data points at the end of each component's segment.
    buffer = (0, 0)
    select_antenna = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['antenna.scio'], buffer) == 1)
    select_res100 = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['res100.scio'], buffer) == 1)
    select_res50 = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['res50.scio'], buffer) == 1)
    select_short = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['short.scio'], buffer) == 1)
    select_noise = (pzt.shrink_flag(prizm_data['100MHz']['switch_flags']['noise.scio'], buffer) == 1)
    
    # The above filters can now be used to separate the spectra for each component and polarization.
    
    # Polarization 0:
    antenna_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_antenna]
    res100_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_res100]
    res50_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_res50]
    short_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_short]
    noise_data_pol0 = prizm_data['100MHz']['pol0.scio'][select_noise]
    
    # Polarization 1:
    antenna_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_antenna]
    res100_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_res100]
    res50_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_res50]
    short_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_short]
    noise_data_pol1 = prizm_data['100MHz']['pol1.scio'][select_noise]
    
    
    # Here we plot the Polarization 0 channels for rach component as an example.
    
    # Antenna:
    plt.figure(1, figsize=(10, 10))
    plt.title('Pol0 (Antenna)')
    plt.imshow(np.log10(antenna_data_pol0), vmin=4.5, vmax=8.5)
    plt.show()
    
    # Short:
    plt.figure(2, figsize=(20, 10))
    plt.title('Pol0 (Short)')
    plt.imshow(np.log10(short_data_pol0), vmin=4.5, vmax=8.5)
    plt.show()
    
    # Res 50:
    plt.figure(3, figsize=(20, 10))
    plt.title('Pol0 (Res 50)')
    plt.imshow(np.log10(res50_data_pol0), vmin=4.5, vmax=8.5)
    plt.show()
    
    # Res 100:
    plt.figure(4, figsize=(20, 10))
    plt.title('Pol0 (Res 100)')
    plt.imshow(np.log10(res100_data_pol0), vmin=4.5, vmax=8.5)
    plt.show()
    
    # Noise:
    plt.figure(5, figsize=(20, 10))
    plt.title('Pol0 (Noise)')
    plt.imshow(np.log10(noise_data_pol0), vmin=4.5, vmax=8.5)
    plt.show()
    
