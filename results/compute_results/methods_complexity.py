
# %%
""" 
Computing markers of EEG signal complexity
Created on 2025.02.25. Updated on 2025.11.05, 2026.02.16
Author: Lina Jeantin.

The following methods :
- Permutation entropy: epochs_permutation_entropy(), _define_symbols(), _symb_permutation()
- Kolmogorov complexity: epochs_komplexity(), _symb_kolmogorov(), _komplexity_python()
were updated and adapted from NICE tools v. 0.1.dev1 (https://nice-tools.github.io/nice/),
licensed under the GNU Affero General Public License version 3.
Original publication to cite: Engemann D.A.*, Raimondo F.*, King JR. et al., Robust EEG-based cross-site 
and cross-protocol classification of states of consciousness. 
Brain. Vol 141 (11), 3160–3178, doi:10.1093/brain/awy251

The following methods are custom functions built on top of MNE-python:
- epochs_power_analysis()
- epochs_spectral_entropy()
- compute_complexity()

"""

# %%

import math
import numpy as np
from tqdm.auto import tqdm
import mne
from mne.utils import logger, _time_mask
from itertools import permutations
from scipy.signal import butter, filtfilt
import time
import zlib
import warnings

# %%

##########################################
# Filters
highpass = 0.5
lowpass = 45

# Epochs
epoch_length = 2.0
overlap_dur = 0.0

## Parameters for spectral analysis & markers of EEG complexity
param_psds = {
    "n_fft": 200,
    "n_overlap": None,
    "n_per_seg": 100,
}

param_freq = {
    "delta": {"min": highpass, "max": 4, "tau": 64}, 
    "theta": {"min": 4, "max": 8, "tau": 32},
    "alpha": {"min": 8, "max": 12, "tau": 16},
    "beta": {"min": 12, "max": 30, "tau": 8},
    "gamma": {"min": 30, "max": lowpass, "tau": 4},
}

kernel_pe = 3  # Size of sub-vectors (order of permutation)
n_bins_komp=32 # number of bins used for the symbolic transform

# %%

# =======================================================================================
# ==================================   SPECTRUM ANALYSIS  ===============================
# =======================================================================================


def epochs_power_analysis(
    epochs,
    freq_bands,
    highpass=0.1,
    lowpass=45.0,
    n_fft=250,
    n_overlap=125,
    n_per_seg=250,
    eps=1e-12,
):
    """
    Computes PSDs (Welch) and returns:
      - psd_full: power per epoch, channel and frequency value (n_epochs, n_channels, n_freqs)
      - total_power: total power in all frequency bands (n_epochs, n_channels)
      - band_power_raw: raw power for a given frequency band (n_epochs, n_channels, n_bands). Returns a stacked array.
      - band_power_rel: normalized power of a given band by the total power (n_epochs, n_channels, n_bands). Returns a stacked array.
            ## To get the raw and normalized power of a given band :
                    b = np.where(band_names == "alpha")[0][0]
                    alpha_rel = band_power_rel[:, :, b]   # (n_epochs, n_chans)
                    alpha_raw = band_power_raw[:, :, b]   # (n_epochs, n_chans)
      - band_names: (n_bands,)  (array of str)
      - freqs: (n_freqs,)
    """
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, seeg=True, exclude='bads')
    if n_overlap is None:
        n_overlap = n_fft/2

    psd_full, freqs = epochs.compute_psd(
        method='welch',
        fmin=highpass,
        fmax=lowpass,
        picks=picks,
        proj=True,
        n_jobs=-1,
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_per_seg=n_per_seg,
        average='mean',
        window='hamming',
        verbose=None
    ).get_data(return_freqs=True)  # (n_epochs, n_channels, n_freqs), (n_freqs,)

    total_power = psd_full.sum(axis=2)  # (n_epochs, n_channels)

    # Band masks
    band_names = np.array(list(freq_bands.keys())) # array of str
    band_masks = []
    for name in band_names:
        fb = freq_bands[name]
        fmin, fmax = float(fb["min"]), float(fb["max"])
        mask = (freqs >= fmin) & (freqs < fmax)
        if not np.any(mask) and fmax >= freqs[-1]:
            mask = (freqs >= fmin) & (freqs <= fmax + eps) # eps if last band is highest freq
        band_masks.append(mask)

    # per band
    band_power_raw = np.stack(
        [psd_full[:, :, m].sum(axis=2) if np.any(m) else np.zeros(psd_full.shape[:2])
         for m in band_masks],
        axis=2
    )

    # normalise
    denom = np.maximum(total_power[..., None], eps)  # (n_epochs, n_channels, 1)
    band_power_norm = band_power_raw / denom

    return psd_full, total_power, band_power_raw, band_power_norm, band_names, freqs


# %%

# =======================================================================================
# ==================================   SPECTRAL ENTROPY   ===============================
# =======================================================================================

def epochs_spectral_entropy(
    epochs,
    fmin=0.1,
    fmax=45.0,
    method='welch',
    n_fft=250,
    n_overlap=125,
    n_per_seg=250,
    normalize=False,
):
    """
    Compute spectral entropy (one value per epoch and channel).
    H = -np.sum(p * np.log(p), axis=2) with p = normalised PSDs (psd / total_power, shape: n_epochs, n_channels, n_freqs)
    - If 'normalize' = False: entropy in 'nats'
    - If normalize: entropy in [0, 1] (divide by H_max = ln(F)) with F= number of freq bins
    Returns: np.array of shape (n_epochs, n_channels).
    """
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, seeg=True, exclude='bads')
    if n_overlap is None:
        n_overlap = n_fft/2
        
    psd = epochs.compute_psd(
        method=method,
        fmin=fmin,
        fmax=fmax,
        n_fft=n_fft,
        n_overlap=n_overlap,
        n_per_seg=n_per_seg,
        picks=picks,
        verbose=None
    ).get_data(return_freqs=False) # PSD (n_epochs, n_channels, n_freqs)

    eps = 1e-15
    psd_sum = psd.sum(axis=2, keepdims=True)                 # n_epochs, n_channels, 1
    p = psd / np.maximum(psd_sum, eps)                       # n_epochs, n_chanels, n_freqs

    # Shannon's Entropy
    H = -np.sum(p * np.log(p + eps), axis=2)                 # n_epochs, n_channels

    # Option: normalise in [0, 1]
    if normalize:
        F = psd.shape[2]                                     # F = number of freq. bins
        H_max = np.log(max(F, 1))                            # ln(F)
        H_norm = H / max(H_max, eps)                         # [0, 1]
        return H, H_norm
    else:
        return H, None


# %%

# =======================================================================================
# ==================================  PERMUTATION ENTROPY ===============================
# =======================================================================================

def epochs_permutation_entropy(epochs, kernel, tau_ms, tmin=None, tmax=None,
                      backend='python', method_params=None):
    """
    Compute Permutation Entropy (PE)

    Parameters:
        epochs : instance of mne.Epochs. The epochs on which to compute the PE (one value of PE per epoch and channel).
        kernel : int. The number of samples to use to transform to a symbol (size of the subvector).
        tau : int. The number of samples left between the ones that defines a symbol.
        backend : {'python',}.
        method_params : Set other parameters, such as 'filter_freq' if desired: defined as 80/tau or as samplingfreq/(kernel*tau)

    Returns:
        pe: np.array object. A np.array of shape (n_channels, n_epochs) containing PE values for each channel and epoch.
        sym: np.array object
            A np.array of shape (n_channels, temporal_length, n_epochs) containing values obtained after the symbolic transform,
            where 'temporal_length' corresponds to the length of the subvectors for the symbolic transform,
            computed by : temporal_length = ntimes - tau * (kernel-1),
            and ntimes is (epoch_length * sampling_freq), the number of time points in an epoch.
    """
    if method_params is None:
        method_params = {}

    freq = float(epochs.info['sfreq']) # Float
    tau_samples = max(1, int(round((tau_ms / 1000.0) * freq))) # tau must be an integer

    picks = mne.pick_types(epochs.info, meg=True, eeg=True, seeg=True) # meg & eeg & seeg
    data = epochs.get_data()[:, picks, ...]  
    n_epochs = len(data)
    data = np.hstack(data)

    if 'filter_freq' in method_params:
        filter_freq = method_params['filter_freq']
    else:
        filter_freq = np.double(freq) / kernel / tau_samples
    
    nyq = freq / 2.0
    if not (0 < filter_freq < nyq):
        filter_freq = 0.45 * nyq # Reasonable roll-off: 0.45*Nyquist

    logger.info('Filtering  at %.2f Hz' % filter_freq)
    b, a = butter(6, 2.0 * filter_freq / np.double(freq), 'lowpass') # applies a Butterworth filter of order 6 at the desired freq

    fdata = np.transpose(np.array(
        np.split(filtfilt(b, a, data), n_epochs, axis=1)), [1, 2, 0])

    time_mask = _time_mask(epochs.times, tmin, tmax) # if tmin and tmax are specified, 
                                                     # applies a temporal mask to use times between tmin and tmax
    fdata = fdata[:, time_mask, :] # take data within the temporal mask

    if backend == 'python':
        logger.info("Performing symbolic transformation")
        with tqdm(total=1, desc="Performing symbolic transformation (Python Backend)", leave=True) as pbar:
            sym, count = _symb_permutation(fdata, kernel, tau_samples) # performs the symbolic transform in sub-vectors with kernel and tau parameters
            pbar.update(1)
        pe = np.nan_to_num(-np.nansum(count * np.log(count + 1e-25), axis=1)) # Computes PE (Shannon's formula)
    else:
        raise ValueError('backend %s not supported for PE'
                         % backend)
    nsym = math.factorial(kernel) # n symbolic, total number of possible permutations, corresponding to kernel! (factorial)
    pe = pe / np.log(nsym) # normalizing by dividing by log(n!) (factorial kernel)
    return pe, sym

# Define symbols for symbolic transform
def _define_symbols(kernel): 
    '''
    Defines all the possible permutations of n=kernel points. Attributes a unique ID to each permutation.

    Parameters:
        kernel: int. The number of samples to use to transform to a symbol (size of the subvector)

    Returns:
        result: list. List of symbols ordered by their identifiers.
    '''
    result_dict = dict()
    total_symbols = math.factorial(kernel) # total number of permutations
    cursymbol = 0 # counter used to assign IDs to symbols
    for perm in permutations(range(kernel)): # generates all possible permutations for indices 0 to kernel-1
        order = ''.join(map(str, perm)) # converts permutations into a string
        if order not in result_dict:
            result_dict[order] = cursymbol # associates the current symbol (order) to a unique ID (cursymbol)
            cursymbol = cursymbol + 1 # increments cursymbol to a new value
            result_dict[order[::-1]] = total_symbols - cursymbol # associates the reverse order (order[::-1]) to the symmetrical ID
                                                                 # in the range of symbols. Eg, if order = "021" and 
                                                                 # total_symbols=6, "021" is set to cursymbol 0 and "120" is set to 
                                                                 # total_symbols−cursymbol=5
    result = []
    for v in range(total_symbols):
        for symbol, value in result_dict.items():
            if value == v:
                result += [symbol]
    return result # returns a list of symbols ordered by their identifiers


# Performs symbolic transformation accross 1st dimension
def _symb_permutation(data, kernel, tau):
    """
    Compute the symbolic transform: attributes a symbol to each subvector of size 'kernel'.

    Parameters:
        data: np.array. Obtained from epochs.get_data(), of shape (n_epochs, n_channels, n_times)
            After np.hstack(data), shape (n_channels, n_epochs*n_times)
        kernel : int. The number of samples to use to transform to a symbol (size of the subvector)
        tau : int. The number of samples left between the ones that defines a symbol.

    Returns:
        signal_sym : np.array
            An array containing he symbols associated to the subvectors extracted from the signal, 
            of shape (n_channels, temporal_length, n_epochs)

        (count / signal_sym_shape[1]) : np.array
            An array containing the normalised probabilities of the distribution of symbols, 
            of shape (n_channels, n_symbols, n_epochs) where n_symbols = kernel! (facorial)
    """
    if tau < 1:
        raise ValueError("tau must be >= 1 (in samples)")
    
    symbols = _define_symbols(kernel) ## calls the list of symbols generated by _define_symbols()
    dims = data.shape

    temporal_length = data.shape[1] - tau * (kernel - 1)
    if temporal_length <= 0:
        raise ValueError(
            f"tau*({kernel}-1) = {tau*(kernel-1)} >= n_times_sel={data.shape[1]}. "
            "Decrease tau_ms or kernel."
        )

    signal_sym_shape = list(dims)
    signal_sym_shape[1] = data.shape[1] - tau * (kernel - 1) # each subvector uses n = kernel points with a 'tau' spacing
                                                             # Reduces the number of symbols of tau x (kernel-1)
    signal_sym = np.zeros(signal_sym_shape, np.int32) # creates an array to store symbols associated to each subvector

    count_shape = list(dims)
    count_shape[1] = len(symbols)
    count = np.zeros(count_shape, np.int32) # array to store the probability for each symbol

    for k in tqdm(range(signal_sym_shape[1]), desc="Performing symbolic transform"):
        subsamples = range(k, k + kernel * tau, tau)
        ind = np.argsort(data[:, subsamples], 1)
        signal_sym[:, k, ] = np.apply_along_axis(
            lambda x: symbols.index(''.join(map(str, x))), 1, ind)

    count = np.double(np.apply_along_axis(
        lambda x: np.bincount(x, minlength=len(symbols)), 1, signal_sym)) # bincount creates a histogram of symbol index occurrences
                                                                # minlength guarantees that each symbol has a count (even if it is 0).

    return signal_sym, (count / signal_sym_shape[1])

# %%

# =======================================================================================
# =================================  KOLMOGOROV COMPLEXITY ==============================
# =======================================================================================


def epochs_komplexity(epochs, nbins, tmin=None, tmax=None,
                              backend='python', method_params=None):
    """
    Compute Kolmogorov complexity (K) (one value per epoch and channel)

    Parameters:
        epochs : instance of mne.Epochs. The epochs on which to compute the Kolmogorov complexity.
        nbins : int. Number of bins to use for symbolic transformation (usually chosen between 8 and 128)
        method_params : dictionary. Overrides default parameters for the backend used.
            OpenMP specific {'nthreads'}
        backend : {'python'}

    Returns:
        komp: np.array
            A np.array of shape (n_channels, n_epochs) containing Kolmogorov complexity values for each channel and epoch,
            obtained via the 'k' returned by _komplexity_python() (see below).
    """
    picks = mne.pick_types(epochs.info, meg=False, eeg=True, seeg=True, exclude='bads')

    if method_params is None:
        method_params = {}

    data = epochs.get_data()[:, picks if picks is not None else Ellipsis] # if 'picks' is not defined, Ellipsis selects all data from
                                                                          # all dimensions.
    time_mask = _time_mask(epochs.times, tmin, tmax) ## If Kolmogorov is to be computed for a given tmin-tmax interval for each epoch
    data = data[:, :, time_mask]
    logger.info("Running KolmogorovComplexity")

    if backend == 'python':
        start_time = time.time()
        komp = _komplexity_python(data, nbins) ## Computes complexity (see below)
        elapsed_time = time.time() - start_time
        logger.info("Elapsed time {} sec".format(elapsed_time))
    else:
        raise ValueError('backend %s not supported for KolmogorovComplexity'
                         % backend)
    return komp


def _symb_kolmogorov(signal, nbins):
    """
    Computes the symbolic transform of the signal.
    
    Parameters:
        signal : np.array. A one-dimensional array containing the signal amplitudes.
        nbins : int. Number of bins to use for symbolic transformation (usually between 8 and 128)
    
    Returns:
        osignal.tostring(): str. A string of symbols representing signal amplitudes ordered in bins.
    """
    if nbins < 2:
        raise ValueError("nbins must be >= 2")

    ssignal = np.sort(signal) # sorts the signal amplitude values (ascending order) to identify the bins' limits
    items = signal.shape[0] # number of points in the signal
    first = int(items / 10) # position corresponding to the lower 10% (removes 10% inferior outliers)
    last = items - first if first > 1 else items - 1 # position corresponding to 90% of the signal (removes 10% superior outliers)
    lower = ssignal[first] # lower = value corresponding to the 10th percentile of the signal's amplitudes
    upper = ssignal[last] # 90th percentile of the signal's amplitudes

    if upper <= lower: ### warning if flat EEG segments
        warnings.warn("Flat signal detected in symbolic transform. Returning empty byte string.")
        return b""
    bsize = (upper - lower) / float(nbins)
    if bsize <= 0:
        return b""


    osignal = np.zeros(signal.shape, dtype=np.uint8) # Empty array of the same size as the signal, 
                                                     # to store the indices of the corresponding bins.
    maxbin = nbins - 1 # index of the last bin

    for i in range(items):
        tbin = int((signal[i] - lower) / bsize) # Subtracts the lower limit to centre the value in relation to the bins,
                                                # Divide by the size of a bin to determine in which bin the value falls.
        osignal[i] = ((0 if tbin < 0 # if tbin is negative (below 'lower'), attribute '0'
                       else maxbin if tbin > maxbin # if tbin is above 'upper', attribute maxbin,
                       else tbin) # else, keep tbin as it is
                      + ord('A')) # Convert the index into an ASCII character by adding ord(‘A’) 
                                  # (for example, 0 becomes ‘A’, 1 becomes ‘B’, etc.).

    return osignal.tobytes() # converts into a binary representation (byte string)


def _komplexity_python(data, nbins):
    """
    Compute komplexity (K)

    Parameters:
        data: np.array. EEG data obtained from epochs.get_data(). A np.array of shape (n_epochs, n_channels, n_samples)
            where n_samples are the number of time points per epoch = (sampling frequency in Hz * epoch length in s)
        nbins: int
            Number of bins to use for symbolic transformation (usually between 8 and 128)

    Returns:
        k: np.array. A np.array of shape (n_channels, n_epochs) containing Kolmogorov complexity values for each channel and epoch.
    """
    ntrials, nchannels, nsamples = data.shape # returns the shape of the data: ntrials = n_epochs, nchannels, 
                                              # nsamples= number of time points per epoch = (sampling frequency in Hz * epoch length in s)
    k = np.zeros((nchannels, ntrials), dtype=np.float64) # creates an array of shape (n_channels, n_epochs) 
    eps = 1e-12 ## if flat EEG segments
    for trial in range(ntrials):
        for channel in range(nchannels):
            signal = data[trial, channel, :]

            if np.std(signal) < eps or np.allclose(signal, signal[0], rtol=0, atol=eps): ## if flat EEG segments
                k[channel, trial] = np.nan
                warnings.warn(
                    f"Kolmogorov complexity not computed: signal too flat "
                    f"(trial {trial}, channel {channel}). NaN returned."
                )
                continue
            string = _symb_kolmogorov(signal, nbins) # converts the signal of one epoch and one channel into 
                                                                      # a symbolic transform with _symb_kolmogorov()
                                                                      # Returns a string of ASCII characters for each epoch and channel.
            if len(string) == 0:
                k[channel, trial] = np.nan
                continue
            cstring = zlib.compress(string) # compress the string by using zlib library, returns a compressed string 'cstring'
            k[channel, trial] = float(len(cstring)) / float(len(string)) # the Kolmogorov complexity value is computed by dividing
                                                                         # the length of the compressed string by 
                                                                         # the length of the original, uncompressed string.

    return k # shape (n_channels, n_epochs)


# %%

# =======================================================================================
# =================================   MASK BAD DATA SPANS  ==============================
# =======================================================================================

def get_invalid_epochs(raw, epochs, bad_labels=['del', 'rejected', 'flat', 'volt']):
    sfreq = raw.info['sfreq']
    invalid_epochs = []
    for idx, epoch_event in tqdm(enumerate(epochs.events), desc="Identifying invalid epochs", total=len(epochs)):
        epoch_start = epoch_event[0] / sfreq
        epoch_end = epoch_start + (epochs.times[-1] - epochs.times[0])

        for annot in raw.annotations:
            annot_start, annot_end = annot['onset'], annot['onset'] + annot['duration']
            if annot['description'] in bad_labels and annot_start <= epoch_end and annot_end >= epoch_start:
                invalid_epochs.append(idx)
                break
    return invalid_epochs


def mask_invalid_epochs(data, invalid_epochs):
    data[invalid_epochs] = np.nan
    return data


# %%

# =======================================================================================
# =================================    ORCHESTRA FUNCTION  ==============================
# =======================================================================================

def compute_complexity(epochs, 
                        invalid_epochs,
                        compute_power =         True,
                        compute_spect_entrop =  True, 
                        compute_perm_entrop =   True,
                        compute_komp =          True,
                        highpass=0.1, 
                        lowpass=80.,
                        f_bands = None,
                        param_psds = None,
                        kernel_pe=3,
                        tmin=None, # for PE and komp
                        tmax=None, # for PE and komp
                        backend_pe='python',
                        n_bins_komp=32,
                        backend_komp='python',
):
    if f_bands is None:
        f_bands = {
            "delta": {"min": highpass, "max": 4,  "tau": 64.0},
            "theta": {"min": 4,       "max": 8,  "tau": 32.0},
            "alpha": {"min": 8,       "max": 12, "tau": 16.0},
            "beta":  {"min": 12,      "max": 30, "tau": 8.0},
            "gamma": {"min": 30,      "max": lowpass, "tau": 4.0},
        }
    if param_psds is None:
        param_psds = {"n_fft": 250, "n_overlap": 125, "n_per_seg": 250}
    

    results={}

    if compute_power:
        psd_full, total_power, band_power_raw, band_power_norm, band_names, freqs = epochs_power_analysis(
            epochs,
            freq_bands=f_bands,
            highpass=highpass,
            lowpass=lowpass,
            eps=1e-12,
            ** param_psds
        )

        results['psd_full'] = mask_invalid_epochs(psd_full, invalid_epochs)
        results['total_power'] = mask_invalid_epochs(total_power, invalid_epochs)

        name_to_idx = {name: i for i, name in enumerate(band_names)}
        for band in band_names:
            b = name_to_idx[band]
            results[f'rpsd_{band}'] = band_power_raw[:, :, b] # (n_epochs, n_chans)
            results[f'npsd_{band}'] = band_power_norm[:, :, b] # (n_epochs, n_chans)
            # Mask invalid epochs
            results[f'rpsd_{band}'] = mask_invalid_epochs(results[f'rpsd_{band}'], invalid_epochs)
            results[f'npsd_{band}'] = mask_invalid_epochs(results[f'npsd_{band}'], invalid_epochs)


    if compute_spect_entrop:
        H, _ = epochs_spectral_entropy(
            epochs,
            fmin=highpass,
            fmax=lowpass,
            method='welch',
            normalize=False,
            ** param_psds,
        )
        results['spectral_entropy'] = mask_invalid_epochs(H, invalid_epochs) # n_epochs, n_channels
    
    if compute_perm_entrop:
        for band_name, params in f_bands.items():
            pe, _ = epochs_permutation_entropy(
                epochs=epochs,
                kernel=kernel_pe,
                tau_ms=params['tau'],
                tmin=tmin,
                tmax=tmax,
                backend=backend_pe,
                method_params=None,
            )
            pe_epxch = pe.T  # transpose to (n_epochs, n_channels)
            results[f'pe_{band_name}'] = mask_invalid_epochs(pe_epxch, invalid_epochs)
    
    if compute_komp:
        komp = epochs_komplexity(epochs, nbins=n_bins_komp, tmin=tmin, tmax=tmax,
                                            backend=backend_komp, method_params=None)
        results['komp'] = mask_invalid_epochs(komp.T, invalid_epochs) # transpose to (n_epochs, n_channels)
    
    return results

# %%
