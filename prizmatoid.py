# NumPy
import numpy as np

# Scio (https://github.com/sievers/scio)
import scio

# OS Control and Directory Manipulation
import os, glob

# Date and Time
import time, datetime



def expand_flag(flag, interval):
    """ Expands each flagged sample featuring in the `flag` field by a given sample `interval = (b, a)`. In other words, each flagged sample is expanded to also include those `b` of samples occurring before it, as well a those `a` samples occurring after it.
    
    Args:
        flag: the flag field, i.e., an (n,)-dimensional NumPy array containing entries with value `1` for flagged samples, and `0` for non-flagged samples.
        interval: a tuple of the form `(b, a)` where `b` and `a` are integers specifying how many samples before (`b`) and after (`a`) each flagged sample should be included in the new flag field.
    
    Returns:
        An (n,)-dimensional NumPy array containing encapsulating a new flag field for which a number samples before (`b`) and after (`a`) each flagged sample in `flag` are now also flagged.
    
    Example(s):
        >>> flag = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0.])
        >>> expand_flag(flag, interval=(1, 1))
        >>> array([0., 0., 1., 1., 1., 0., 1., 1., 1.])
    
    """

    # Extracts the entries of `interval`.
    b = interval[0]
    a = interval[1]
    
    # Locates the indiced associated with the flagged samples in `flag`.
    indices = np.where(np.asarray(flag) > 0)[0]
    
    # Create a `new_flag` field as a copy of the input `flag`.
    new_flag = np.asarray(flag).copy()
    
    # Extends each flagged entry of `new_flag` by `b` preceeding samples, and by `a` succeeding samples.
    for index in indices:
        new_flag[max(index - b, 0):min(index + a + 1, new_flag.size - 1)] = 1
    
    # Return the expanded `new_flag`.
    return new_flag



def ctime_from_timestamp(dates, format='%Y%m%d_%H%M%S'):
    """ Given a time stamp (or list of time stamps) in human-friendly format, with default being 'YYYYMMDD_HHMMSS', converts to `datetime` object and calculate the corresponding ctime. (Here ctime stands for the number of seconds since 1970/1/1).
    
    Args:
        dates: a string or a list of strings containing the time stamp(s) in desired text `format`.
        format: the input format of the time stamp(s) contained in `dates`.
    
    Returns:
        A float, or list of floats, containing the ctime for each input time stamp.
    
    Example(s):
        >>> ctime_from_timestamp('19700101_000000')
        >>> 0.0
        >>> ctime_from_timestamp(['19700101_000000', '19700201_000000'])
        >>> [0.0, 2678400.0]
    
    """

    # Sets the reference time used in the definition of ctime.
    reference_time = datetime.datetime(1970, 1, 1)
    
    # Checks whether `dates` is a single string, or a list of strings. For the former, a single value is returned, while for the latter a list of values associated with each entry in `dates` is returned. (Notice that we use `str` since the type `basestring` is no longer available in Python 3).
    if isinstance(dates, str):
        return (datetime.datetime.strptime(dates, format) - reference_time).total_seconds()
    else:
        return [(datetime.datetime.strptime(entry, format) - reference_time).total_seconds() for entry in dates]



def timestamp_from_ctime(ctimes, format='%Y%m%d_%H%M%S'):
    """ Converts a ctime value (or list of ctime values) into human-friendly time stamp in the default format 'YYYYMMDD_HHMMSS'. (Here ctime stands for the number of seconds since 1970/1/1).
    
    Args:
        ctimes: a (list of) int(s) or float(s) containing the ctime(s) to be converted.
        format: the output format of the time stamp(s) associated with the input `ctimes`.
    
    Returns:
        A string, or list of strings, containing the time stamp(s) for each input ctime.
    
    Example(s):
        >>> timestamp_from_ctime(0.0)
        >>> '19700101_000000'
        >>> timestamp_from_ctime([0.0, 2678400.0])
        >>> ['19700101_000000', '19700201_000000']
    
    """

    # Checks whether `ctimes` is a single int or float, or a list of ints or floats. For the former, a single string is returned, while for the latter a list of strings associated with each entry in `ctimes` is returned.
    if isinstance(ctimes, (int, float)):
        return str(datetime.datetime.utcfromtimestamp(ctimes).strftime(format))
    else:
        return [str(datetime.datetime.utcfromtimestamp(entry).strftime(format)) for entry in ctimes]



def dir_from_ctime(first_ctime, second_ctime, dir_parent, n_digits=5):
    """ Retrieves the list of corresponding subdirectories labeled according to an initial and final ctime. This function assumes that the parent directory `dir_parent` has two levels of subdirectories: the first of these being composed of directories labeled by the first `n_digits` of the reference ctime, and the second level being composed of directories labeled by the full 10-digit reference ctime stamp (See the Example below for a concrete case).
    
    Args:
        first_ctime: an int or float specifying the first reference ctime stamp.
        second_ctime:  an int or float specifying the second reference ctime stamp.
        dir_parent: a string specifying the parent directory.
        n_digits:  number of ctime digits labeling the first level of subdirectories.
    
    Returns:
        Returns a list of strings specifying the subdirectories of interest according to the input time range.
    
    Example(s):
        >>> dir_from_ctime(1555335072, 1555336372, dir_parent='..')
        >>> ['../15553/1555335072', ..., '../15553/1555336372']
    
    """

    # Initializes the list of directories `dir_list` which will collect the desired paths.
    dir_list = []
    
    # Checks whether `first_ctime` < `second_ctime`. If not, the values of these inputs are swapped.
    if first_ctime > second_ctime:
        first_ctime, second_ctime = second_ctime, first_ctime
    
    # Creates a NumPy array containing all possible ctimes from `first_ctime` to `second_ctime`, and converts the result to a list.
    ctimes = np.arange(first_ctime, second_ctime + 1, 1).tolist()
    
    # Generates all existent two-level subdirectories by trying each entry of `ctimes`. The results are concatenated into `dir_list`.
    for entry in ctimes:
        dir_list += glob.glob(dir_parent + '/' + str(entry)[0:n_digits] + '/' + str(entry))
    
    # Returns a list of strings specifying the subdirectories of interest according to the input time range.
    return dir_list



def read_scio_file(dirs, file_name, verbose=True):
    """ Reads '.scio' files located in a given list of directories. (This function is largely equivalent to `prizmtools.read_pol_fast`).
    
    Args:
        dir: a list of strings specifying the directories where the '.scio' files of interest are stored.
        file_name: a string in the format '*.scio' specifying the name of the file of interest.
        verbose: a boolean parameter which instructs the function to output messages as the data is read when `True`, or to output no messages when `False`.
    
    Returns:
        A NumPy array containing the information encapsulated in all files named `file_name` stored in the directories `dirs`.
    
    """

    # Checks whether `dirs` is a single string, instead of a list of strings. If that is the case, it is transformed into a list with the provided string as its single entry. This will guarantee compability with the rest of the code that follows.
    if isinstance(dirs, str):
        dirs = [dirs]
    
    # Generates a list which concatenates the string `file_name` to all string entries in the the input list of directories `dirs`.
    file_list = [d + '/' + f for d, f in zip(dirs, [file_name]*len(dirs))]
    
    # Reads the '.scio' file into `scio_data_list`, which is a list of NumPy arrays with each array corresponding to a different entry in `dirs`. This operation is timed.
    read_start = time.time()
    scio_data_list = scio.read_files(file_list)
    read_end = time.time()
    
    # Verbose message.
    if verbose:
        print('`read_scio_file`: operation `scio.read_files` lasted ', read_end - read_start, 's.')
    
    # Checks whether any file has not been found and/or read, therefore appearing in `scio_data_list` as a `None` entry. The indices associated with such files are stored in the list `indices`.
    indices = []
    for index, entry in enumerate(scio_data_list):
    # The entries in `scio_data_list` can be either NumPy arrays or `None`. Since these objects are very different from each other, it is hard to make simple comparisons between them without running into errors. Thus, we must use 'try/except' to prevent the function from hiccuping at this step.
        try:
            # If files could not be found and/or read, the index associated with such files are stored in `indices`, and the path to that files is printed along with a warning message.
            if entry == None:
                indices.append(index)
                print('Could not find and/or read file: ' + dirs[index] + '/' + file_name)
        except ValueError:
            continue
    
    # Keeps only those entries of `scio_data_list` which are not `None`, i.e., keeps all elements which do not feature in the list of `indices` generated above.
    scio_data_list = [entry for index, entry in enumerate(scio_data_list) if index not in indices]
    
    # Attempts to stack the entries of `scio_data_list` into a single NumPy array `scio_data` using `numpy.vstack`. In case this operation fails, it means `scio_data_list` is empty, in which case a warning message is printed.
    try:
        scio_data = np.vstack(scio_data_list)
    except ValueError:
        scio_data = []
        print('No files named `' + file_name + '` could be found and/or read.')
    
    # Returns the `scio_data`.
    return scio_data



def read_raw_file(dirs, file_name, verbose=True, dtype='float64'):
    """ Reads '.raw' files located in a given list of directories. (This function is largely equivalent to `prizmtools.read_field_many_fast`).
    
    Args:
        dir: a list of strings specifying the directories where the '.raw' files of interest are stored.
        file_name: a string in the format '*.raw' specifying the name of the file of interest.
        verbose: a boolean parameter which instructs the function to output messages as the data is read when `True`, or to output no messages when `False`.
        dtype: the desired data type to be returned, defaulted to be 'float64'.
    
    Returns:
        A NumPy array containing the information encapsulated in all files named `file_name` stored in the directories `dirs`.
    
    """

    # Checks whether `dirs` is a single string, instead of a list of strings. If that is the case, it is transformed into a list with the provided string as its single entry. This will guarantee compability with the rest of the code that follows.
    if isinstance(dirs, str):
        dirs = [dirs]
    
    # Reads the '.raw' file into `raw_data_list`, which is a list of NumPy arrays with each array corresponding to a different entry in `dirs`. This operation is timed. In case a file has not been found and/or read, its corresponding indix is stored in the list `indices`.
    indices = []
    raw_data_list = []
    read_start = time.time()
    for index, dir in enumerate(dirs):
        try:
            raw_data_list.append(np.fromfile(dir + '/' + file_name, dtype=dtype))
        except:
            indices.append(index)
    read_end = time.time()
    
    # Verbose message.
    if verbose:
        print('`read_raw_file`: operation `numpy.fromfile` lasted ', read_end - read_start, 's.')
    
    # If files could not be found and/or read, the index associated with such file is stored in `indices` are used to print their path along with a warning message.
    for index in indices:
        print('Could not find and/or read file: ' + dirs[index] + '/' + file_name)
    
    # Attempts to stack the entries of `raw_data_list` into a single NumPy array `raw_data` using `numpy.hstack`. In case this operation fails, it means `raw_data_list` is empty, in which case a warning message is printed.
    try:
        raw_data = np.hstack(raw_data_list)
    except ValueError:
        raw_data = []
        print('No files named `' + file_name + '` could be found and/or read.')
    
    # Returns the `raw_data`.
    return raw_data



def read_prizm_data(first_ctime, second_ctime, dir_top, subdir_100='data_100MHz', subdir_70='data_70MHz', subdir_switch='switch_data', read_100=True, read_70=True, read_switch=True, verbose=False):
    """ Reads PRIZM data within a specified time range. The reading of both frequency channels, as well as temperature and switch data are enabled by default.
    
    Args:
        first_ctime: an int or float specifying the first reference ctime stamp.
        second_ctime:  an int or float specifying the second reference ctime stamp.
        dir_top = a string containing the top level directory where the data is stored.
        subdir_100 = a string specifying the subdirectory within the top level directory where the 100 MHz channel data is stored.
        subdir_70 = a string specifying the subdirectory within the top level directory where the 70MHz channel data is stored.
        subdir_switch = a string specifying the subdirectory within the top level directory where the switch state data is stored.
        read_100: a boolean parameter which determines whether or not the 100 MHz channel data must be read.
        read_70: a boolean parameter which determines whether or not the 70 MHz channel data must be read.
        read_switch: a boolean parameter which determines whether or not the switch state data must be read.
        verbose: a boolean parameter which instructs the function to output messages as the data is read when `True`, or to output no messages when `False`.
    
    Returns:
        A dictionary containing all PRIZM data found in the directories corresponding to the input ctimes. A typical dictionary returned by this function would have the following structure.
        {'70MMz': {'pol0.scio': numpy.array, 'pol1.scio': numpy.array, 'cross_real.scio': numpy.array, 'cross_imag.scio': numpy.array},
         '100MHz': {'pol0.scio': numpy.array, 'pol1.scio': numpy.array, 'cross_real.scio': numpy.array, 'cross_imag.scio': numpy.array},
         'switch_data: {'antenna.scio': numpy.array, 'res100.scio': numpy.array, 'res50.scio': numpy.array, 'short.scio': numpy.array}}
    
    """
    
    # Initializes the dictionary which will hold the data.
    prizm_data = {}
    
    # Lists the typical '*.scio' and '*.raw' file names and their respective data types.
    scio_files = ['pol0.scio', 'pol1.scio', 'cross_real.scio', 'cross_imag.scio']
    raw_files = [('acc_cnt1.raw','int32'), ('acc_cnt2.raw','int32'), ('fft_of_cnt.raw','int32'), ('fft_shift.raw','int64'), ('fpga_temp.raw','float'), ('pi_temp.raw','int32'), ('sync_cnt1.raw','int32'), ('sync_cnt2.raw','int32'), ('sys_clk1.raw','int32'), ('sys_clk2.raw','int32'), ('time_sys_start.raw','float'), ('time_sys_stop.raw','float')]
    switch_files = ['antenna.scio','res100.scio','res50.scio','short.scio']
    
    # Primary Data:
    # Checks whether `read_100` and `read_70` are `True`. If so, their respective keys are stored in the list `antennas`, and also created as entries for the `prizm_data` dictionary. The input subdirectories `subdir_100` and `subdir_70` are also stored in the `subdirs` dictionary for future manipulation.
    antennas = []
    subdirs = {}
    if read_100:
        antennas.append('100MHz')
        subdirs['100MHz'] = subdir_100
    if read_70:
        antennas.append('70MHz')
        subdirs['70MHz'] = subdir_70
    
    # Verbose message.
    if (verbose and len(antennas) > 0):
        print('Reading primary data from the ', antennas, 'atennas.')
    
    # Reads the primary data products for the 70 MHz and 100 Mhz antennas.
    for antenna in antennas:
        prizm_data[antenna] = {}
        dirs = dir_from_ctime(first_ctime, second_ctime, dir_top + '/' + subdirs[antenna])
    
        # Reads all '.scio' files in `dirs` whose names match the entries in `scio_files`. The results are stored in the appropriate antenna dictionary entry of `prizm_data` with key given by the file name.
        for file_name in scio_files:
            prizm_data[antenna][file_name] = read_scio_file(dirs, file_name, verbose=verbose)
    
        # Reads all '.raw' files in `dirs` whose names match the entries in `raw_files`. The results are stored in the appropriate antenna dictionary entry of `prizm_data` with key iven by the file name.
        for file_name, dtype in raw_files:
            prizm_data[antenna][file_name] = read_raw_file(dirs, file_name, verbose=verbose, dtype=dtype)
    
    # Auxiliary Data:
    # Checks whether `read_switch` is `True`. If so, the key `switch` is added to the `prizm_data` dictionary, creates the list of directories where the switch data is located, and proceeds to read the data.
    if read_switch:
        prizm_data['switch'] = {}
        dirs = dir_from_ctime(first_ctime, second_ctime, dir_top + '/' + subdir_switch)
    
        # Verbose message.
        if verbose:
            print('Reading the switch auxiliary data.')
    
        # Reads all '.scio' files in `dirs` whose names match the entries in `switch_files`. The results are stored as dictionaries in `prizm_data['switch']`, with keys given by the file names being read.
        for file_name in switch_files:
            prizm_data['switch'][file_name] = read_scio_file(dirs, file_name, verbose=verbose)
    
    # Returns the `prizm_data` found in the given time range.
    return prizm_data



def add_switch_flags(prizm_data, antennas=['70MHz', '100MHz']):
    """ Adds a 'switch_flags' entry for the `antennas` featuring in the input `prizm_data` dictionary. These new entries are based on the auxiliary switch data contained in that same dictionary which flags the PRIZM data as either coming from: the antenna, the 100 Ohm resistor, the short, or the 50 Ohm resistor.
    
    Args:
        prizm_data: a dictionary containing all PRIZM data structured according to the output of the function `read_prizm_data`.
        atennas: a list containing the antennas for flag generation.
    
    Returns:
        The input dictionary with an additional entry with key 'switch_flags' for each antenna listed in `antennas`. The new entry contains a dictionary with keys 'antenna.scio', 'res100.scio', 'short.scio', and 'res50.scio' associated with NumPy arrays encapsulating the flags for each of these PRIZM components. A typical output returned by this function would have the following structure.
        {'70MMz': { ..., 'switch_flags': {'antenna.scio': numpy.array, 'res100.scio': numpy.array, 'short.scio': numpy.array, 'res50.scio': numpy.array}},
         '100MHz': { ..., 'switch_flags': {'antenna.scio': numpy.array, 'res100.scio': numpy.array, 'short.scio': numpy.array, 'res50.scio': numpy.array}},
         'switch_data: {'antenna.scio': numpy.array, 'res100.scio': numpy.array, 'res50.scio': numpy.array, 'short.scio': numpy.array}}
    
    """

    # Recovers the keys in `prizm_data['switch']`.
    switch_files = prizm_data['switch'].keys()
    
    # Adds flags for each antenna.
    for antenna in antennas:
    
        # Makes sure the input dictionary contains entries for antenna(s) of interest. An error message is printed if that information is missing.
        if antenna not in prizm_data.keys():
            print('`add_switch_flags`: error, the data for the ' + antenna + ' antenna could not be found.')
            continue
    
        # Makes sure the input dictionary contains the timestamp data. An error message is printed if that information is missing.
        if len(prizm_data[antenna]['time_sys_start.raw']) == 0:
            print('`add_switch_flags`: error, no timestamp data was found for the ' + antenna + ' antenna.')
            continue
        
        # Initializes the dictionary entry which will store the flags.
        prizm_data[antenna]['switch_flags'] = {}
    
        # Collects the start and stop times stored in `prizm_data[antenna]['time_sys_start.raw']`.
        start_time = prizm_data[antenna]['time_sys_start.raw'][0]
        stop_time = prizm_data[antenna]['time_sys_stop.raw'][-1]
    
        # Generates the flags and adds them to `prizm_data`.
        for file_name in switch_files:
            times = prizm_data['switch'][file_name]
    
            # Initializes the NumPy array `flag` which will be used in the flags generation below.
            flag = np.zeros_like(prizm_data[antenna]['time_sys_start.raw'], dtype='int')
    
            # Artificially adds endpoints in case those are missing. The starting endpoint is characterized by `np.array([[1.0, start_time]])`, while the final endpoint is characterized by `np.array([[0.0, end_time]])`.
            if len(times) > 0 and times[0,0] == 0.0:
                starting_endpoint = np.array([[1.0, start_time]])
                times = np.append(starting_endpoint, times, axis=0)
            if len(times) > 0 and times[-1,0] == 1.0:
                final_endpoint = np.array([[0.0, stop_time]])
                times = np.append(times, final_endpoint, axis=0)
                
            # Slices the data into chunks delimited in time by the entries in `times`. These are used to create a fiter `filter_chunk` which picks only data matching the chunk under consideration.
            for chunk_start, chunk_end in zip(times[:-1], times[1:]):
                chunk_filter = np.where((prizm_data[antenna]['time_sys_start.raw'] >= chunk_start[1]) & (prizm_data[antenna]['time_sys_stop.raw'] <= chunk_end[1]))[0]
    
                # If the current element (antenna, resistance, or short) is active for the chunk under consideration (i.e., `chunk_start[0] == 1.0`), the `flag` is assigned the value `1` in that chunk.
                if chunk_start[0] == 1.0:
                    flag[chunk_filter] = np.ones(len(chunk_filter), dtype='int')
    
            # Adds flags to `prizm_data`.
            prizm_data[antenna]['switch_flags'][file_name] = flag
    
    return



#def add_flag(scihi_dat, antennas=['70', '100']):
#    """Add fast sampled data quality flag for specified antenna(s).
#    Spectrum is marked as bad if accumulation counter changed during
#    the read, FFT overflow was triggered, and...other criteria TBD.
#
#    - scihi_dat: SCI-HI data dictionary with appropriate antenna and switch entries
#    - antennas: list of antennas for flag generation
#
#    Creates fast sampled bit field flag with the following values:
#    bit 0 = accumulation counter changed
#    bit 1 = FFT overflow triggered
#    ...others to be added later.
#
#    Flag is added to the SCI-HI data dictionary as 'flag'.
#    """
#
#    bvals = {'acc_cnt':0,
#             'fft_of_cnt':1}
#
#    for antenna in antennas:
#        flag = np.zeros_like(scihi_dat[antenna]['time_start'], dtype='int')
#        if len(scihi_dat[antenna]['time_start']) == 0:
#            print('add_switch_flag: error, no timestamp data from antenna', antenna)
#            continue
#        # Accumulation counter check
#        inds = np.where( scihi_dat[antenna]['acc_cnt2']-scihi_dat[antenna]['acc_cnt1'] != 0.0 )[0]
#        flag[inds] = ( flag[inds] | np.array([2**bvals['acc_cnt']]*len(inds), dtype='int') )
#        # FFT overflow check
#        inds = np.where( scihi_dat[antenna]['fft_of_cnt'] != 0.0 )[0]
#        flag[inds] = ( flag[inds] | np.array([2**bvals['fft_of_cnt']]*len(inds), dtype='int') )
#
#        scihi_dat[antenna]['flag'] = flag
#
#    return
