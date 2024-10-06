# Import libraries
import numpy as np
import pandas as pd
from obspy import read
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import os
import sys

from concurrent.futures import ThreadPoolExecutor

from scipy import signal
from matplotlib import cm

from obspy.signal.filter import highpass
from obspy.signal.trigger import classic_sta_lta, plot_trigger, trigger_onset
from obspy.signal.invsim import cosine_taper

from scipy.fft import fft, fftfreq

tr = 0

width_divider = 4001

def findPeak(tr_times, filtered, path):

    # Sampling frequency of our trace
    df = tr.stats.sampling_rate

    # How long should the short-term and long-term window be, in seconds? # TODO: Make dynamic
    sta_len = 500
    lta_len = 6000

    # # Pad the beginning of the arrays with zeros
    # # Create negative time values for padding on the left
    # pad_times_left = np.linspace(-lta_len / df, -1 / df, lta_len)  # df is the sampling frequency
    # pad_data_left = np.ones(lta_len)  # Pad the data with zeros on the left

    # print(len(pad_data_left))

    # # Pad the left side of the arrays with negative times and zeros
    # tr_times_padded = np.concatenate((pad_times_left, tr_times))
    # tr_data_padded = np.concatenate((pad_data_left, filtered))

    # #Set the minimum frequency
    # # minfreq = 0.5
    # # maxfreq = 1.0

    # # To better see the patterns, we will create a spectrogram using the scipy function
    # # It requires the sampling rate, which we can get from the miniseed header as shown a few cells above
    f, t, sxx = signal.spectrogram(tr.data, df)

    t = tr_times[0] + t

    # # Pre-process the trace by applying a taper to avoid edge effects
    # taper = cosine_taper(len(tr_data_padded), 0.05)
    # tr_data_filtered = tr_data_padded * taper

    tr_data_filtered = filtered.copy()
    # tr_data_filtered = tr_data_padded

    # High-pass filter to remove low-frequency noise
    # tr_data_filtered = highpass(tr_data_filtered, 0.1, df, corners=2, zerophase=True)

    # Run Obspy's STA/LTA to obtain a characteristic function
    # cft = classic_sta_lta(tr_data_filtered, int(sta_len * df), int(lta_len * df))
    # cft /= np.max(cft)

    # cft[:int(lta_len*df)] = filtered[:int(lta_len*df)]

    # Threshold for peak detection based on noise level
    noise_level = np.mean(tr_data_filtered)
    noise_level = 1.2 * noise_level
    peak_threshold = noise_level  # Adjust this value based on data
    print(f'Noise level: {noise_level}')
    print(f'Peak threshold: {peak_threshold}')

    print("Min. width for peak detection = ", int((tr_times[-1] / width_divider) * df), "sec.")

    # Minimum width requirement (number of timesteps)
    min_timesteps_above_threshold = int((tr_times[-1] / width_divider) * df)  # adaptive to different durations for the datasets

    # Manual peak detection (without using find_peaks)
    valid_peaks = []
    on_off_candidates = []  # Store the candidate "on" and "off" windows

    i = 0
    while i < len(tr_data_filtered):

        if tr_data_filtered[i] > peak_threshold:
            
            # Start of a potential peak
            peak_start = i 
            # Move forward while the value stays above the threshold
            
            while i < len(tr_data_filtered) and tr_data_filtered[i] > peak_threshold:
                i += 1
            
            # End of the peak
            peak_end = i
            
            print(f"Potential peak: {tr_times[peak_start]} - {tr_times[peak_end - 1]}")

            # Check if the width of the peak meets the minimum requirement
            width = tr_times[peak_end - 1] - tr_times[peak_start]
            
            print(f"Width: {width} (width in seconds: {np.round(width / df, 2)}) | peak amplitude: {np.round(tr_data_filtered[peak_start:peak_end].max(), 2)}")
            
            if width >= min_timesteps_above_threshold:
                
                # print("Valid with = ", peak_end, peak_start, width, min_timesteps_above_threshold)

                # Add this as a valid peak and window candidatecfFalset
                peak_idx = np.argmax(tr_data_filtered[peak_start:peak_end]) + peak_start
                valid_peaks.append(peak_idx)
                on_off_candidates.append((peak_start, peak_end, tr_data_filtered[peak_idx]))
        
        else:
            
            i += 1

    print(f'Number of valid peaks: {len(valid_peaks)}')

    # If we have valid peaks, find the one with the highest amplitude
    if valid_peaks:
        # Sort the candidates by peak amplitude (the third element in on_off_candidates)
        on_off_candidates_sorted = sorted(on_off_candidates, key=lambda x: tr_data_filtered[x[0]:x[1]].max(), reverse=True)
        highest_valid_peak = on_off_candidates_sorted[0]  # The one with the highest peak

        # Now we use this peak to calculate the dynamic thresholds
        max_amplitude = highest_valid_peak[2]  # Highest peak amplitude from valid peaks
    else:
        max_amplitude = 0  # Default value if no valid peaks found
        on_off_candidates_sorted = [(0,0,0)]

    # Set dynamic trigger thresholds based on the highest valid peak amplitude
    on_threshold = max_amplitude * 0.9  # Set 'on' trigger threshold
    off_threshold = max_amplitude * 0.1  # Set 'off' trigger threshold

    # Create a new empty list to store only the final "on" and "off" triggers
    on_off_final = []

    # Find the onset and offset of triggers with the new thresholds but only for the highest peak
    for start, end, _ in on_off_candidates_sorted:

        if start == end:
            on_off_final.append((0,0))
            print("No clear seismic activity detected!")
            

        elif max_amplitude == tr_data_filtered[start:end].max():
            on_off_final.append((start, end))  # Only keep the highest peak triggers

    # Relative times for plotting
    # tr_times = np.arange(0, len(tr_data_filtered)) / df

    on_off_final = on_off_final[0]

    display = True

    if display == True:

        # Initialize the dynamic figure
        fig, ax = plt.subplots(3, 1, figsize=(10,10), sharex = True)

        print("debug = ",tr_times[0], tr_times[-1])

        # Plot the time series and spectrogram   
        ax[0].plot(tr_times, tr.data)
        ax[0].set_title('Raw signal of file: ' + path, fontweight='bold')

        # Make the plot pretty
        ax[0].set_xlim([min(tr_times),max(tr_times)])
        ax[0].set_ylabel('Velocity (m/s)', fontweight = 'bold')
        # ax[0].set_xlabel('Time (s)')

        # Plot "on" and "off" lines for all valid peaks, but with one label for each type
        on_line_drawn = False  # Control flag to add the 'Trig. On' label only once
        off_line_drawn = False  # Control flag to add the 'Trig. Off' label only once

        # if len(on_off_candidates_sorted) > 1:
        #     on_off_candidates_sorted = on_off_candidates_sorted[0]

        for start, end, _ in on_off_candidates_sorted:
            if not on_line_drawn:
                ax[0].axvline(x=tr_times[start], color='magenta', linestyle='--', label='Trig. On')
                on_line_drawn = True  # Ensures the label is added only once
            else:
                ax[0].axvline(x=tr_times[start], color='magenta', linestyle='--')  # Add line without label

            if not off_line_drawn:
                ax[0].axvline(x=tr_times[end], color='green', linestyle='--', label='Trig. Off')
                off_line_drawn = True  # Ensures the label is added only once
                break
            else:
                ax[0].axvline(x=tr_times[end], color='green', linestyle='--')  # Add line without label

        ax[0].legend()

        # Spectrogram plot
        #vals = ax[1].pcolormesh(t, f, sxx, cmap=cm.BuPu)
        vals = ax[1].pcolormesh(t, f, sxx, cmap=cm.gist_stern)
        # ax[1].set_xlim(min(tr_times), max(tr_times))
        ax[1].set_title('Spectrogram of the raw data', fontweight='bold' )
        ax[1].set_ylabel('Frequency (Hz)', fontweight='bold')
        cbar = plt.colorbar(vals, ax=ax[1], orientation='horizontal', location = 'top', pad = 0.2)
        cbar.set_label('Power ((m/s)^2/sqrt(Hz))', fontweight='bold')

        # Add the same "on" and "off" lines for the spectrogram
        for start, end, _ in on_off_candidates_sorted:
            ax[1].axvline(x=tr_times[start], color='magenta', linestyle='--')
            ax[1].axvline(x=tr_times[end], color='green', linestyle='--')
            break

        # Plot the filtered signal
        ax[2].plot(tr_times, filtered)
        ax[2].set_title('Filtered Signal', fontweight='bold' )
        ax[2].set_xlabel('Time [s]', fontweight='bold')
        ax[2].set_ylabel('Amplitude (Normed)', fontweight = 'bold')
        # ax[2].set_xlim(min(tr_times),max(tr_times))
        y = np.full_like(tr_times, noise_level)
        ax[2].plot(tr_times, y, color = 'lightgrey')

        # Add the same "on" and "off" lines for the filtered signal
        for start, end, _ in on_off_candidates_sorted:
            ax[2].axvline(x=tr_times[start], color='magenta', linestyle='--')
            ax[2].axvline(x=tr_times[end], color='green', linestyle='--')
            break

        # # Plot the seismogram
        # ax[3].plot(tr.times(), cft)
        # ax[3].set_title('cfd')

        # # Plot only the highest-amplitude "on" and "off" triggers (first entry in on_off_final)
        # if len(on_off_final) > 0:
        #     ax[3].axvline(x=tr_times[on_off_final[0]], color='red', linestyle='--', label='Trig. On')
        #     ax[3].axvline(x=tr_times[on_off_final[1]], color='purple', linestyle='--', label='Trig. Off')

        # # Set plot limits and labels
        # ax[3].set_xlim([min(tr_times), max(tr_times)])
        # ax[3].set_xlabel('Time (s)')
        # ax[3].set_ylabel('Amplitude')

        # # Add legend
        # ax[3].legend()

        # Show the plot
        # plt.show()

        plt.subplots_adjust(hspace=0.25)  # Increase the value for more padding

        plt.savefig('plotsB12Luna_ML/' + path.removesuffix('.mseed') + '.png')  # Saves as PNG with high resolution
        plt.clf()

def readMINIseed(path, detected_time):
    
    # read in miniseed
    st = read(path) 

    # This is how you get the data and the time, which is in seconds
    tr = st.traces[0].copy()
    
    first = int((detected_time - 2500) *  tr.stats.sampling_rate)
    last = int((detected_time + 7500) *  tr.stats.sampling_rate)

    print(first, last)

    print(len(tr.data), len(tr.times()))

    tr.data = tr.data[first:last]   
    tr.times()[first:last]

    tr_times = [0] * len(tr.times())
    for i in range(0, len(tr_times)):
        tr_times[i] = tr.times()[i] + first

    print(tr_times[0])

    tr_filtered = tr.copy()

    # ge tthe point where there are the most frequencies
    N = len(tr_filtered)  # Number of sample points
    fft_values = fft(tr_filtered)  # FFT of the signal
    fft_magnitude = np.abs(fft_values) / N  # Normalized amplitude
    frequencies = fftfreq(N, 1 / tr.stats.sampling_rate)  # Frequency axis

    freqmin = frequencies[np.argmax(fft_magnitude)] * 2.3

    tr_filtered = tr_filtered.filter('highpass', freq = freqmin).copy()
    tr_filtered = np.abs(tr_filtered)

    return tr, tr_times, tr_filtered

def confFilter(directory_path, path, detected_times):

    global tr
    
    print(path)

    tr, tr_times, tr_filtered = readMINIseed(directory_path + path, detected_time)

    tr_filtered = signal.medfilt(tr_filtered, kernel_size = 201)
    tr_filtered /= np.max(tr_filtered) # Norm 
    
    # # Set the sampling rate based on the time constant and scale factor
    sampling_rate = tr.stats.sampling_rate

    # 4000 / 84000
    ratio = 0.0476

    settling_time = int(tr.times()[-1] * ratio)
    peak_time = settling_time / 1.9  # hand tuned To be improved

    # Step 1: Calculate the natural frequency (omega_n) and damping ratio (zeta)
    # Using equations for peak time and settling time
    omega_n = np.pi / (peak_time * np.sqrt(1 - (4 / (settling_time * np.pi))**2))
    zeta = 4 / (settling_time * omega_n)

    # Step 2: Define the second-order system's transfer function (numerator = [omega_n^2], denominator = [1, 2*zeta*omega_n, omega_n^2])
    num = [omega_n**2]
    den = [1, 2*zeta*omega_n, omega_n**2]

    # # Calculate the number of points based on the adjusted sampling rate
    num_points = int((settling_time) / sampling_rate) + 1

    # Generate the time array
    t = np.linspace(0, settling_time, num_points)

    #Generate the impulse response
    t, imp_response = signal.impulse((num , den), T = t)

    filtered = signal.correlate(tr_filtered, imp_response, mode='same')
    
    # filtered = signal.medfilt(filtered, kernel_size = 301)
    
    filtered /= np.max(filtered)

    findPeak(tr_times, filtered, path)

    # plotFiltered(tr, tr_filtered, filtered, path, imp_response, t)

    return tr, tr_filtered, filtered

def plotFiltered(tr, tr_filtered, filtered, path, imp_response, t):

    # Initialize the dynamic figure
    fig, ax = plt.subplots(5, 1, figsize=(10,2))
    
    # Plot trace
    ax[0].plot(tr.times(), tr.data/ np.max(tr.data))
    # Mark detection
    # ax[0].axvline(x = arrival, color='red',label='Rel. Arrival')
    # ax[0].legend(loc='upper left')
    # Make the plot pretty
    ax[0].set_xlim([min(tr.times()),max(tr.times())])
    ax[0].set_ylabel('Velocity (m/s)')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_title(f'{path}', fontweight='bold')
    
    # Plot the single triangular pulse
    ax[1].plot(tr.times(), tr_filtered)
    ax[1].set_title('Filtered Signal')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlim([min(tr.times()),max(tr.times())])

    # Plot the single triangular pulse
    ax[2].plot(tr.times(), filtered)
    ax[2].set_title('filtered with correlation')
    ax[2].set_xlabel('Time [s]')
    ax[2].set_ylabel('Amplitude')
    ax[2].set_xlim([min(tr.times()),max(tr.times())])

    # Plot the single triangular pulse
    ax[4].plot(t, imp_response)
    ax[4].set_title('Single Triangular Pulse Signal')
    ax[4].set_xlabel('Time [s]')
    ax[4].set_ylabel('Amplitude')
    ax[4].set_xlim([min(tr.times()),max(tr.times())])
 
    plt.show()

    # plt.savefig('plotsB12Luna/' + path.removesuffix('.mseed') + '.png')  # Saves as PNG with high resolution
    # plt.clf()

def getWindow(catalogue_directory, folder_path):

    return


if __name__ == "__main__":

    # directory_path = "./data/mars/test/data/"
    directory_path = "./data/lunar/test/data/"
    folder_path = "S12_GradeB"

    catalogue_directory = './data/lunar/test/catalogs_ML/'

    csv_file = f'{catalogue_directory}{folder_path}.csv'
    data_cat = pd.read_csv(csv_file)

    files_data_with_hit = []

    for i in range(0, len(data_cat['filename'])):
        files_data_with_hit.append([data_cat['filename'][i] + ".mseed", data_cat['time_rel(sec)'][i]])

    # List all files in the directory
    # files = os.listdir(directory_path)

    # # Filter out directories, only keep files
    # files_in_directory = [f for f in files if f.endswith('.mseed') and os.path.isfile(os.path.join(directory_path, f))]
    # files_in_directory.sort()

    # # Get the number of files and their names
    # num_files = len(files_in_directory)
    # file_names = files_in_directory

    # "./data/lunar/test/data/S12_GradeB/" -> [4] -> endgegner

    # file_names = 'xa.s12.00.mhz.1972-05-19HR00_evid00228.mseed'

    for file_name, detected_time in files_data_with_hit:

        confFilter(directory_path + folder_path + "/", file_name, detected_time)
        print("")




    


