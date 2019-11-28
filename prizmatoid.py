# NumPy
import numpy as np

# Scio (https://github.com/sievers/scio)
import scio

# OS Control and Directory Manipulation
import os
import glob

# Date and Time
import time
from datetime import datetime



def expand_flag(flag, interval):
    """ Expands each flagged sample featuring in the `flag` field.

    Given the `interval = (b, a)`, each flagged entry in `flag` is extended so
    as to also flag those `b` preceding entries, as well as those `a` succeeding
    entries.

    Args:
        flag: the flag field, i.e., an (n,)-dimensional NumPy array containing
            integers of value `1` for flagged entries, and `0` for non-flagged
            entries.
        interval: a tuple of the form `(b, a)` where `b` and `a` are integers
            specifying how many entries before (`b`) and after (`a`) each
            flagged entries should be assigned the value `1` in the new flag
            field.

    Returns:
        An (n,)-dimensional NumPy array making up a new flag field which, in
        addition to containing all flagged entries present in the input `flag`,
        also flags a number of entries occurring before (`b`) and after (`a`)
        each of those initially-flagged entries.

    Example(s):
        >>> flag = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0.])
        >>> expand_flag(flag, interval=(1, 1))
        >>> array([0., 0., 1., 1., 1., 0., 1., 1., 1.])
    """

    # Extracts the entries of `interval`.
    b = interval[0]
    a = interval[1]

    # Locates the indices associated with the flagged samples in `flag`.
    indices = np.where(np.asarray(flag) > 0)[0]

    # Create a `new_flag` field as a copy of the input `flag`.
    new_flag = np.asarray(flag).copy()

    # Extends each flagged entry of `new_flag` by `b` preceeding samples, and by
    # `a` succeeding samples.
    for index in indices:
        new_flag[max(index - b, 0):min(index + a + 1, new_flag.size - 1)] = 1

    # Return the expanded `new_flag`.
    return new_flag


def ctime_from_timestamp(dates, format='%Y%m%d_%H%M%S'):
    """ Calculates the ctime associated with a given timestamp.

    Calculates the ctime(s) associated with a timestamp (or list of timestamps).
    The default timestamp format is set to 'YYYYMMDD_HHMMSS'.

    Args:
        dates: a string (or a list of strings) containing the timestamp(s) which
            one wishes to convert to ctime(s).
        format: the input format of the timestamp(s) given by (or listed in)
            `dates`. (See the 'datetime' module documentation for more details
            on the syntax of timestamp formatting).

    Returns:
        A list of floats containing the ctimes corresponding to each input
        timestamp.

    Example(s):
        >>> ctime_from_timestamp('19700101_000000')
        >>> 0.0
        >>> ctime_from_timestamp(['19700101_000000', '19700201_000000'])
        >>> [0.0, 2678400.0]
    """

    # Checks whether `dates` is a single string. If that is the case, it is
    # converted into a list with the provided string as its single entry. This
    # will guarantee compability with the code that follows.
    if isinstance(dates, str):
        dates = [dates]

    # Sets the reference time `ref_time` used in the definition of ctime (i.e.,
    # the number of seconds since 1970/1/1).
    ref_time = datetime(1970, 1, 1)

    # Generates the `ctimes` list by subtracting `ref_time` from each entry in
    # `dates` and expressing the result in units of seconds.
    ctimes = [
        (datetime.strptime(entry, format) - ref_time).total_seconds()
        for entry in dates
        ]

    # Returns the `ctimes` list.
    return ctimes


def timestamp_from_ctime(ctimes, format='%Y%m%d_%H%M%S'):
    """ Obtains the timestamp associated with a given ctime value.

    Converts a ctime value (or list of ctime values) into human-friendly
    timestamp(s). The default timestamp format is set to 'YYYYMMDD_HHMMSS'.

    Args:
        ctimes: a int or a float (or a list of such) containing the ctime(s)
            which one wishes to convert to timestamp format.
        format: the output format of the timestamp(s) associated with the input
            `ctimes`.

    Returns:
        A list of strings containing a timestamp for each input ctime.

    Example(s):
        >>> timestamp_from_ctime(0.0)
        >>> '19700101_000000'
        >>> timestamp_from_ctime([0.0, 2678400.0])
        >>> ['19700101_000000', '19700201_000000']
    """

    # Checks whether `ctimes` is a single integer or float. If that is the case,
    # it is converted into a list with the provided value as its single entry.
    # This will guarantee compability with the code that follows.
    if isinstance(ctimes, (int, float)):
        ctimes = [ctimes]

    # Generates the `dates` list containing the timestamps of interest.
    dates = [
             str(datetime.utcfromtimestamp(entry).strftime(format))
             for entry in ctimes
             ]

    # Returns the `dates` list.
    return dates


def dir_from_ctime(first_ctime, second_ctime, dir_parent, n_digits=5):
    """ Retrieves a list of subdirectory paths pointing to the data of interest.

    Retrieves a list of subdirectory paths pointing to the PRIZM data which has
    been recorded between the `first_ctime` and the `second_ctime`. This
    function assumes that the parent directory `dir_parent` has two levels of
    subdirectories: the first of these being composed of directories labeled by
    the first `n_digits` of a reference ctime, and the second level being
    composed of directories labeled by the full 10-digit reference ctime.

    Args:
        first_ctime: an int or float specifying the first reference ctime.
        second_ctime: an int or float specifying the second reference ctime.
        dir_parent: a string specifying the data's parent directory.
        n_digits: an integer specifying the number of ctime digits labeling the
            first level of subdirectories.

    Returns:
        Returns a list of strings specifying the subdirectories containing PRIZM
        data which has been recorded within the input time range.

    Example(s):
        >>> dir_from_ctime(1555335072, 1555336372, dir_parent='..')
        >>> ['../15553/1555335072', ..., '../15553/1555336372']
    """

    # Initializes the list of directories `dir_list` which will collect the
    # desired paths.
    dir_list = []

    # Checks whether `first_ctime` < `second_ctime`. If not, the values of these
    # inputs are swapped.
    if first_ctime > second_ctime:
        first_ctime, second_ctime = second_ctime, first_ctime

    # Lists all subdirectories in the first level of the directory structure,
    # i.e., all subdirectories labeled by the first `n-digits` of a reference
    # ctime.
    first_level = os.listdir(dir_parent)

    # Cleans any non-numeric entries in `first_level` and sorts the remaining
    # entries.
    first_level = [
                   entry
                   for entry in first_level
                   if entry.isnumeric()
                   ]
    first_level.sort()

    # Loops over the entries of the first level of subdirectories in order to
    # identify the subdirectories in the second level.
    for first_level_entry in first_level:
        # If the `first_level_entry` ctime dos not fall within the time
        # range defined by the input `first_ctime` and `second_ctime`, skip
        # to the next one.
        if (int(first_level_entry) < int(str(first_ctime)[:n_digits])
            or int(first_level_entry) > int(str(second_ctime)[:n_digits])):
            continue
        # Else, checks what are the subdirectories in `first_level_entry`.
        else:
            # Lists all subdirectories in the second level of the directory
            # structure.
            second_level = os.listdir(dir_parent + '/' + first_level_entry)
            
            # Cleans any non-numeric entries in `second_level` and sorts
            # the remaining entries.
            second_level = [
                            entry
                            for entry in second_level
                            if entry.isnumeric()
                            ]
            second_level.sort()
            
            # Creates `second_level_num` by converting all entries of
            # `second_level` to their corresponding numerical values.
            # This makes numerical comparisons with `first_ctime` and
            # `second_ctime` easier.
            second_level_num = np.asarray(second_level, dtype='float')
            
            # Converts `second_level` to a NumPy array of strings to
            # facilitate its slicing/manipulation below.
            second_level = np.asarray(second_level)
            
            # Estipulates the condition for picking directories of interest,
            # i.e., those which fall within the input ctime range.
            condition = np.logical_and(second_level_num - first_ctime >= 0,
                                       second_level_num - second_ctime <= 0)

            # Selects the directories of interest by applying `condition` to
            # `second_level`. Then stores the result in `dir_list`.
            dir_select = [
                          dir_parent
                          + '/' + first_level_entry
                          + '/' + second_level_entry
                          for second_level_entry in second_level[condition]
                          ]
            dir_list += dir_select

    # Sorts and returns a list of strings specifying the subdirectories of
    # interest according to the input time range.
    dir_list.sort()
    return dir_list


def read_scio_file(dirs, file_name, verbose=True):
    """ Reads '.scio' files located in a given list of directories.

    Looks for files with the given `file_name` in the input list of directories
    `dirs`. If the file has been located in the provided directory, the function
    attempts to read it. In case the file cannot be found and/or read, an error
    message is printed. All files which have been successfully located and read
    are stacked and returned as a single NumPy array.
    (This function is largely equivalent to `prizmtools.read_pol_fast`).

    Args:
        dir: a list of strings specifying the directories where the '.scio'
            files of interest are stored.
        file_name: a string in the format '*.scio' specifying the name of the
            file of interest.
        verbose: a boolean parameter which instructs the function to output
            messages as the data is read when `True`, or to output no messages
            when `False`.

    Returns:
        A NumPy array containing the information encapsulated in all files named
        `file_name` stored in the directories `dirs`. If no files with the input
        `file_name` can be found and/or read, an empty NumPy array is returned.
    """

    # Checks whether `dirs` is a single string. If that is the case, it is
    # converted into a list with the provided string as its single entry. This
    # guarantees compability with the rest of code that follows.
    if isinstance(dirs, str):
        dirs = [dirs]

    # Generates a list which appropriately concatenates the string `file_name`
    # to all string entries in the input list of directories `dirs`.
    file_list = [d + '/' + f for d, f in zip(dirs, [file_name]*len(dirs))]

    # Reads the '.scio' file into `scio_data_list`, which is a list of NumPy
    # arrays with each array corresponding to a different entry in `dirs`. This
    # operation is timed in case `verbose = True`.
    read_start = time.time()
    scio_data_list = scio.read_files(file_list)
    read_end = time.time()

    # Verbose message.
    if verbose:
        print(
              '`read_scio_file`: operation `scio.read_files` lasted ',
              read_end - read_start,
              's.',
              )

    # Checks whether any file has not been found and/or read, making it feature
    # in `scio_data_list` as a `None` entry. The indices associated with such
    # files are stored in the list `indices`.
    indices = []
    for index, entry in enumerate(scio_data_list):
        # The entries in `scio_data_list` can be either NumPy arrays or `None`.
        # Since these objects are very different from each other, it is hard to
        # make simple comparisons between them without running into errors.
        # Thus, we must use 'try/except' to prevent the function from hiccuping
        # at this step.
        try:
            # If files could not be found and/or read, the index associated with
            # such files are stored in `indices`, and the path to that files is
            # printed along with a warning message.
            if entry is None:
                indices.append(index)
                print(
                      'Could not find and/or read file: '
                      + dirs[index] + '/' + file_name
                      )
        except ValueError:
            continue

    # Keeps only those entries of `scio_data_list` which are not `None`, i.e.,
    # keeps all elements which do not feature in the list of `indices` generated
    # above.
    scio_data_list = [
                      entry
                      for index, entry in enumerate(scio_data_list)
                      if index not in indices
                      ]

    # Attempts to stack the entries of `scio_data_list` into a single NumPy
    # array `scio_data` using `numpy.vstack`. If this operation fails, it means
    # `scio_data_list` is empty, in which case a warning message is printed and
    # `scio_data` is assigned an empty NumPy array.
    try:
        scio_data = np.vstack(scio_data_list)
    except ValueError:
        scio_data = np.array([])
        print('No files named `' + file_name + '` could be found and/or read.')

    # Returns the `scio_data`.
    return scio_data


def read_raw_file(dirs, file_name, verbose=True, dtype='float64'):
    """ Reads '.raw' files located in a given list of directories.

    Looks for files with the given `file_name` in the input list of directories
    `dirs`. If the file has been located in the provided directory, the function
    attempts to read it. In case the file cannot be found and/or read, an error
    message is printed. All files which have been successfully located and read
    are stacked and returned a single NumPy array.
    (This function is largely equivalent to `prizmtools.read_field_many_fast`).

    Args:
        dir: a list of strings specifying the directories where the '.raw' files
            of interest are stored.
        file_name: a string in the format '*.raw' specifying the name of the
            file of interest.
        verbose: a boolean parameter which instructs the function to output
            messages as the data is read when `True`, or to output no messages
            when `False`.
        dtype: the desired data type to be returned, defaulted to be 'float64'.

    Returns:
        A NumPy array containing the information encapsulated in all files named
        `file_name` stored in the directories `dirs`. If no files with the input
        `file_name` can be found and/or read, an empty NumPy array is returned.
    """

    # Checks whether `dirs` is a single string. If that is the case, it is
    # converted into a list with the provided string as its single entry. This
    # guarantees compability with the rest of code that follows.
    if isinstance(dirs, str):
        dirs = [dirs]

    # Reads the '.raw' file into `raw_data_list`, which is a list of NumPy
    # arrays with each array corresponding to a different entry in `dirs`. This
    # operation is timed in case `verbose = True`. In case a file has not been
    # found and/or read, its corresponding index (i.e., its position in the
    # directory list) is stored in the list `indices`.
    indices = []
    raw_data_list = []
    read_start = time.time()
    for index, dir in enumerate(dirs):
        try:
            raw_data_list.append(
                                 np.fromfile(dir + '/' + file_name, dtype=dtype)
                                 )
        except:
            indices.append(index)
    read_end = time.time()

    # Verbose message.
    if verbose:
        print(
              '`read_raw_file`: operation `numpy.fromfile` lasted ',
              read_end - read_start,
              's.',
              )

    # If files could not be found and/or read, the index associated with such
    # file, which is stored in `indices`, is used to print their path along with
    # a warning message.
    for index in indices:
        print(
              'Could not find and/or read file: '
              + dirs[index] + '/' + file_name
              )

    # Attempts to stack the entries of `raw_data_list` into a single NumPy array
    # `raw_data` using `numpy.hstack`. If this operation fails, it means
    # `raw_data_list` is empty, in which case a warning message is printed and
    # `raw_data` is assigned an empty NumPy array.
    try:
        raw_data = np.hstack(raw_data_list)
    except ValueError:
        raw_data = np.array([])
        print('No files named `' + file_name + '` could be found and/or read.')

    # Returns the `raw_data`.
    return raw_data


def read_prizm_data(first_ctime, second_ctime, dir_top,
        subdir_100='data_100MHz', subdir_70='data_70MHz',
        subdir_switch='switch_data', read_100=True, read_70=True,
        read_switch=True, read_temp=False, verbose=False):
    """ Reads PRIZM data within a specified time range.

    Looks for data files in the subdirectories `subdir_100` and `subdir_70`
    contained in `dir_top` which have been recorded within a given time range.
    The type of data to be read is controled by the boolean parameters
    `read_100`, `read_70`, `read_switch`, and `read_temp`. If the file has been
    successfully located, the function attempts to read it. In case the file
    cannot be found and/or read, an error message is printed. All files which
    have been successfully located and read are stored and returned in a
    dictionary format.

    Args:
        first_ctime: an int or float specifying the first reference ctime stamp.
        second_ctime: an int or float specifying the second reference ctime.
        dir_top = a string containing the top level directory where the data is
            stored.
        subdir_100 = a string specifying the subdirectory within the top level
            directory where the 100 MHz channel data is stored.
        subdir_70 = a string specifying the subdirectory within the top level
            directory where the 70MHz channel data is stored.
        subdir_switch = a string specifying the subdirectory within the top
            level directory where the switch state data is stored.
        read_100: a boolean parameter which determines whether or not the
            100 MHz channel data must be read.
        read_70: a boolean parameter which determines whether or not the 70 MHz
            channel data must be read.
        read_switch: a boolean parameter which determines whether or not the
            switch state data must be read.
        verbose: a boolean parameter which instructs the function to output
            messages as the data is read when `True`, or to output no messages
            when `False`.

    Returns:
        A dictionary containing all PRIZM data found in the directories
        corresponding to the input ctimes. A typical dictionary returned by this
        function would have the following structure.

        {
        '70MHz': {
            'pol0.scio': numpy.array,
            'pol1.scio': numpy.array,
            'cross_real.scio': numpy.array,
            'cross_imag.scio': numpy.array,
            'acc_cnt1.raw': numpy.array,
            'acc_cnt2.raw': numpy.array,
            'fft_of_cnt.raw': numpy.array,
            'fft_shift.raw': numpy.array,
            'fpga_temp.raw': numpy.array,
            'pi_temp.raw': numpy.array,
            'sync_cnt1.raw': numpy.array,
            'sync_cnt2.raw': numpy.array,
            'sys_clk1.raw': numpy.array,
            'sys_clk2.raw': numpy.array,
            'time_sys_start.raw': numpy.array,
            'time_sys_stop.raw': numpy.array,
            },
         '100MHz': {
            'pol0.scio': numpy.array,
            'pol1.scio': numpy.array,
            'cross_real.scio': numpy.array,
            'cross_imag.scio': numpy.array,
            'acc_cnt1.raw': numpy.array,
            'acc_cnt2.raw': numpy.array,
            'fft_of_cnt.raw': numpy.array,
            'fft_shift.raw': numpy.array,
            'fpga_temp.raw': numpy.array,
            'pi_temp.raw': numpy.array,
            'sync_cnt1.raw': numpy.array,
            'sync_cnt2.raw': numpy.array,
            'sys_clk1.raw': numpy.array,
            'sys_clk2.raw': numpy.array,
            'time_sys_start.raw': numpy.array,
            'time_sys_stop.raw': numpy.array,
            },
         'switch': {
            'antenna.scio': numpy.array,
            'res100.scio': numpy.array,
            'res50.scio': numpy.array,
            'short.scio': numpy.array,
            }
        }
    """

    # Initializes the dictionary which will hold the data.
    prizm_data = {}

    # Lists the typical '*.scio' and '*.raw' file names and their respective
    # data types.
    scio_files = [
        'pol0.scio', 'pol1.scio', 'cross_real.scio', 'cross_imag.scio',
        ]
    raw_files = [
        ('acc_cnt1.raw', 'int32'), ('acc_cnt2.raw', 'int32'),
        ('fft_of_cnt.raw', 'int32'), ('fft_shift.raw', 'int64'),
        ('fpga_temp.raw', 'float'), ('pi_temp.raw', 'int32'),
        ('sync_cnt1.raw', 'int32'), ('sync_cnt2.raw', 'int32'),
        ('sys_clk1.raw', 'int32'), ('sys_clk2.raw', 'int32'),
        ('time_sys_start.raw', 'float'),
        ('time_sys_stop.raw', 'float'),
        ('time_rtc_start.raw', 'float'),
        ('time_rtc_stop.raw', 'float'),
        ]
    switch_files = ['antenna.scio', 'res100.scio', 'res50.scio', 'short.scio']
    temp_files = [
        ('temp_100A_bot_lna.raw', 'int32'), ('temp_100_ambient.raw', 'int32'),
        ('temp_100A_noise.raw', 'int32'), ('temp_100A_switch.raw', 'int32'),
        ('temp_100A_top_lna.raw', 'int32'), ('temp_100B_bot_lna.raw', 'int32'),
        ('temp_100B_noise.raw', 'int32'), ('temp_100B_switch.raw', 'int32'),
        ('temp_100B_top_lna.raw', 'int32'), ('temp_70A_bot_lna.raw', 'int32'),
        ('temp_70_ambient.raw', 'int32'), ('temp_70A_noise.raw', 'int32'),
        ('temp_70A_switch.raw', 'int32'), ('temp_70A_top_lna.raw', 'int32'),
        ('temp_70B_bot_lna.raw', 'int32'), ('temp_70B_noise.raw', 'int32'),
        ('temp_70B_switch.raw', 'int32'), ('temp_70B_top_lna.raw', 'int32'),
        ('temp_pi.raw', 'int32'), ('temp_snapbox.raw', 'int32'),
        ('time_pi.raw', 'int32'), ('time_start_therms.raw', 'int32'),
        ('time_stop_therms.raw', 'int32'),
        ]
        
    # Lists the some old '.raw' file names, their respective data types, and
    # their new file nomemclature.
    old_raw_files = [
        ('time_start.raw', 'float', 'time_sys_start.raw'),
        ('time_stop.raw', 'float', 'time_sys_stop.raw'),
        ]

    # Primary Data:
    # Checks whether `read_100` and `read_70` are `True`. If so, their
    # respective keys are stored in the list `antennas`, and also created as
    # entries for the `prizm_data` dictionary. The input subdirectories
    # `subdir_100` and `subdir_70` are also stored in the `subdirs` dictionary
    # for future manipulation.
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
        dirs = dir_from_ctime(first_ctime,
                              second_ctime,
                              dir_top + '/' + subdirs[antenna]
                              )

        # Reads all '.scio' files in `dirs` whose names match the entries in
        # `scio_files`. The results are stored in the appropriate antenna
        # dictionary entry of `prizm_data` with key given by the file name.
        for file_name in scio_files:
            prizm_data[antenna][file_name] = read_scio_file(dirs,
                                                            file_name,
                                                            verbose=verbose)

        # Reads all '.raw' files in `dirs` whose names match the entries in
        # `raw_files`. The results are stored in the appropriate antenna
        # dictionary entry of `prizm_data` with key given by the file name.
        for file_name, dtype in raw_files:
            prizm_data[antenna][file_name] = read_raw_file(dirs,
                                                           file_name,
                                                           verbose=verbose,
                                                           dtype=dtype)

        # Checks whether `prizm_data['time_sys_start.raw']` is empty, which
        # would mean no timestamp information could be found. In that case,
        # attempts to read the timestamp information from those files listed in
        # `old_raw_files`. This step is needed because at some point in 2018 the
        # timestamp files were renamed from 'time_start.raw' and 'time_stop.raw'
        # to 'time_sys_start.raw' and 'time_sys_stop.raw'. Since the timestamp
        # information is essential for any analysis of the PRIZM data, checking
        # whether these older files are available is essential. Notice that
        # despite having different names, the data dictionary keys referring
        # to such files still reflect the more recent file nomenclature in order
        # to keep the resulting `prizm_data` dictionary compatible with other
        # functions defined in this module.
        if len(prizm_data[antenna]['time_sys_start.raw']) == 0:
            # Verbose message.
            if verbose:
                print('Attempting to read the older timestamp files.')

            # Reads all '.raw' files in `dirs` whose names match the entries in
            # `old_raw_files`. The results are stored in the appropriate antenna
            # dictionary entry of `prizm_data` under a key given by the more
            # recent file nomenclature associated with those files.
            for old_file_name, dtype, file_name in old_raw_files:
                prizm_data[antenna][file_name] = read_raw_file(dirs,
                                                               old_file_name,
                                                               verbose=verbose,
                                                               dtype=dtype)

    # Auxiliary Data:
    # Checks whether `read_switch` is `True`. If so, the key `switch` is added
    # to the `prizm_data` dictionary, creates the list of directories where the
    # switch data is located, and proceeds to read the data.
    if read_switch:
        prizm_data['switch'] = {}
        dirs = dir_from_ctime(first_ctime,
                              second_ctime,
                              dir_top + '/' + subdir_switch)

        # Verbose message.
        if verbose:
            print('Reading the switch auxiliary data.')

        # Reads all '.scio' files in `dirs` whose names match the entries in
        # `switch_files`. The results are stored as dictionaries in
        # `prizm_data['switch']`, with keys given by the file names being read.
        for file_name in switch_files:
            prizm_data['switch'][file_name] = read_scio_file(dirs,
                                                             file_name,
                                                             verbose=verbose)

    # Returns the `prizm_data` found in the given time range.
    return prizm_data


def add_switch_flags(prizm_data, antennas=['70MHz', '100MHz']):
    """ Creates a 'switch_flags' entry in a PRIZM data dictionary.

    Adds a 'switch_flags' entry for each of the `antennas` featuring in the
    input `prizm_data` dictionary. These new entries are based on the auxiliary
    'switch' entry contained in that same dictionary which flags portions of the
    PRIZM data as either coming from: the antenna, the 100 Ohm resistor, the
    short, or the 50 Ohm resistor.
    
    Args:
        prizm_data: a dictionary containing all PRIZM data structured according
            to the output of the function `read_prizm_data`.
        antennas: a list containing the antennas for flag generation.
    
    Returns:
        The input dictionary with an additional entry with key 'switch_flags'
        for each antenna listed in `antennas`. The new entry contains a
        dictionary with keys 'antenna.scio', 'res100.scio', 'short.scio', and
        'res50.scio', each storing a NumPy array which flags each of the the
        data as coming from either the antenna, the 100 Ohm resistor, the short,
        or the 50 Ohm resistor. A typical output returned by this function would
        have the following structure.

        {
        '70MHz': {
            'pol0.scio': numpy.array,
            ...,
            'switch_flags': {
                'antenna.scio': numpy.array,
                'res100.scio': numpy.array,
                'short.scio': numpy.array,
                'res50.scio': numpy.array,
                }
            },
         '100MHz': {
            'pol0.scio': numpy.array,
            ...,
            'time_sys_stop.raw',
            'switch_flags': {
                'antenna.scio': numpy.array,
                'res100.scio': numpy.array,
                'short.scio': numpy.array,
                'res50.scio': numpy.array,
                }
            },
         'switch': {
            'antenna.scio': numpy.array,
            'res100.scio': numpy.array,
            'res50.scio': numpy.array,
            'short.scio': numpy.array,
            }
        }
    """

    # Recovers the keys in `prizm_data['switch']`.
    switch_files = prizm_data['switch'].keys()
    
    # Adds flags for each antenna.
    for antenna in antennas:

        # Makes sure the input dictionary contains entries for antenna(s) of
        # interest. An error message is printed if that information is missing.
        if antenna not in prizm_data.keys():
            print(
                  '`add_switch_flags`: the data for the '
                  + antenna
                  + ' antenna could not be found.'
                  )
            continue

        # Makes sure the input dictionary contains the timestamp data. An error
        # message is printed if that information is missing.
        if len(prizm_data[antenna]['time_sys_start.raw']) == 0:
            print(
                  '`add_switch_flags`: no timestamp data was found for the '
                  + antenna
                  + ' antenna.'
                  )
            continue

        # Initializes the dictionary entry which will store the flags.
        prizm_data[antenna]['switch_flags'] = {}

        # Collects the start and stop times stored in
        # `prizm_data[antenna]['time_sys_start.raw']`.
        start_time = prizm_data[antenna]['time_sys_start.raw'][0]
        stop_time = prizm_data[antenna]['time_sys_stop.raw'][-1]

        # Generates the flags and adds them to `prizm_data`.
        for file_name in switch_files:
            # Here `times` contains the ctimes at which the data-taking
            # associated with a given component (antenna, the 100 Ohm resistor,
            # the short, or the 50 Ohm resistor) started and stopped.
            times = prizm_data['switch'][file_name]

            # Initializes the NumPy array `flag` which will be used in the flags
            # generation below.
            flag = np.zeros_like(prizm_data[antenna]['time_sys_start.raw'],
                                 dtype='int')

            # Artificially adds endpoints in case those are missing. The
            # starting endpoint is characterized by
            # `np.array([[1.0, start_time]])`, while the final endpoint is
            # characterized by `np.array([[0.0, end_time]])`.
            if len(times) > 0 and times[0,0] == 0.0:
                starting_endpoint = np.array([[1.0, start_time]])
                times = np.append(starting_endpoint, times, axis=0)
            if len(times) > 0 and times[-1,0] == 1.0:
                final_endpoint = np.array([[0.0, stop_time]])
                times = np.append(times, final_endpoint, axis=0)

            # Takes the ctime data stored in `prizm_data` in preparation for the
            # data chunk selection performed below. Here the NumPy arrays
            # `data_time_start` and `data_time_stop` contain the times at which
            # data (associated with any) given component started and stopped
            # being recorded, respectively.
            data_time_start = prizm_data[antenna]['time_sys_start.raw']
            data_time_stop = prizm_data[antenna]['time_sys_stop.raw']
            
            # Slices the data into chunks delimited in time by the entries in
            # `times`. These are used to create a filter `filter_chunk` which
            # picks only data matching the chunk under consideration.
            for chunk_start, chunk_end in zip(times[:-1], times[1:]):
                condition = np.logical_and(data_time_start >= chunk_start[1],
                                           data_time_stop <= chunk_end[1])
                chunk_filter = np.where(condition)[0]

                # If the current element (antenna, resistance, or short) is
                # active for the chunk under consideration (i.e.,
                # `chunk_start[0] == 1.0`), the `flag` is assigned the value `1`
                # in that chunk.
                if chunk_start[0] == 1.0:
                    flag[chunk_filter] = np.ones(len(chunk_filter), dtype='int')

            # Adds flags to `prizm_data`.
            prizm_data[antenna]['switch_flags'][file_name] = flag

    return


def add_quality_flags(prizm_data, antennas=['70MHz', '100MHz']):
    """ Creates a 'quality_flags' entry in a PRIZM data dictionary.

    Adds a 'quality_flags' entry for each of the `antennas` featuring in the
    input `prizm_data` dictionary. These new entries are based on the
    information contained in the entries 'acc_cnt1.raw', 'acc_cnt2.raw', and
    'fft_of_cnt.raw' of `prizm_data` which monitor the quality of the PRIZM
    data.

    Args:
        prizm_data: a dictionary containing all PRIZM data structured according
            to the output of the function `read_prizm_data`.
        antennas: a list containing the antennas for flag generation.

    Returns:
        The input dictionary with an additional entry with key 'quality_flags'
        for each antenna listed in `antennas`. The new entry contains a
        dictionary with keys 'acc_cnt.raw' and 'fft_of_cnt.raw', each storing a
        NumPy array which flags whether the data quality is acceptable or not.
        A typical output returned by this function would have the following
        structure.

        {
        '70MHz': {
            'pol0.scio': numpy.array,
            ...,
            'quality_flags': {
                'acc_cnt.raw': numpy.array,
                'fft_of_cnt.raw': numpy.array,
                }
            },
         '100MHz': {
            'pol0.scio': numpy.array,
            ...,
            'time_sys_stop.raw',
            'quality_flags': {
                'acc_cnt.raw': numpy.array,
                'fft_of_cnt.raw': numpy.array,
                }
            },
         'switch': {
            'antenna.scio': numpy.array,
            'res100.scio': numpy.array,
            'res50.scio': numpy.array,
            'short.scio': numpy.array,
            }
        }
    """

    # Adds flags for each antenna.
    for antenna in antennas:

        # Makes sure the input dictionary contains entries for antenna(s) of
        # interest. An error message is printed if that information is missing.
        if antenna not in prizm_data.keys():
            print(
                  '`add_quality_flags`: the data for the '
                  + antenna
                  + ' antenna could not be found.'
                  )
            continue

        # Makes sure the input dictionary contains the timestamp data. An error
        # message is printed if that information is missing.
        if len(prizm_data[antenna]['time_sys_start.raw']) == 0:
            print(
                  '`add_switch_flags`: no timestamp data was found for the '
                  + antenna
                  + ' antenna.'
                  )
            continue

        # Initializes the dictionary entry which will store the flags.
        prizm_data[antenna]['quality_flags'] = {}

        # Initializes the reference NumPy array `ref` which will be used to
        # generate flags below.
        ref = np.ones_like(prizm_data[antenna]['time_sys_start.raw'],
                           dtype='int')

        # Checks for consistency in the accumulation as a way of assessing data
        # quality. Flags the consistent chunks with `1` and the non-consistent
        # chunks with `0`.
        counter_consistency = (prizm_data[antenna]['acc_cnt2.raw']
                               - prizm_data[antenna]['acc_cnt1.raw'])
        counter_flag = np.where(counter_consistency == 0.0, ref, ref*0)
        
        # Adds `counter_flag` to `prizm_data` under the new entry 'acc_cnt.raw'.
        prizm_data[antenna]['quality_flags']['acc_cnt.raw'] = counter_flag

        # Checks for FFT overflows as a way of assessing data quality. Flags the
        # acceptable chunks with `1` and the non-acceptable chunks with `0`.
        fft_flag = np.where(prizm_data[antenna]['fft_of_cnt.raw'] == 0.0,
                            ref,
                            ref*0)

        # Adds `fft_flag` to `prizm_data` under the new entry 'fft_of_cnt.raw'.
        prizm_data[antenna]['quality_flags']['fft_of_cnt.raw'] = fft_flag

    return
