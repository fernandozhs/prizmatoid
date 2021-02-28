# NumPy
import numpy as np

# SciPy
from scipy import interpolate

# HealPy
import healpy

# PyGSM (https://github.com/telegraphic/PyGSM)
from pygsm import GSMObserver
from pygsm import GlobalSkyModel

# Scio (https://github.com/sievers/scio)
import scio

# OS Control and Directory Manipulation
import os
import glob

# Date and Time
import time
import suntime
import ephem
from datetime import datetime, timezone, timedelta



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
        new_flag[max(index - b, 0):min(index + a + 1, new_flag.size)] = 1

    # Return the expanded `new_flag`.
    return new_flag


def shrink_flag(flag, interval):
    """ Shrinks each flagged sample featuring in the `flag` field.

    Given the `interval = (b, e)`, each flagged entry in `flag` is reduced by
        unflagging those `b` first entries, as well as those `e` final entries.

    Args:
        flag: the flag field, i.e., an (n,)-dimensional NumPy array containing
            integers of value `1` for flagged entries, and `0` for non-flagged
            entries.
        interval: a tuple of the form `(b, e)` where `b` and `e` are integers
            specifying how many entries in the beginning (`b`) and the end (`e`)
            of each flagged entries should be assigned the value `0` in the new
            flag field.

    Returns:
        An (n,)-dimensional NumPy array making up a new flag field in which
        the original flagged entries present in the input `flag` are reduced in
        length through the unflagging of entries in the beginning (`b`) and the
        end (`e`) of those initially-flagged entries.

    Example(s):
        >>> flag = np.array([0., 0., 1., 1., 1., 0., 1., 1., 1.])
        >>> expand_flag(flag, interval=(1, 1))
        >>> array([0., 0., 0., 1., 0., 0., 0., 1., 0.])
    """

    # Extracts the entries of `interval`.
    b = interval[0]
    e = interval[1]

    # Create a `new_flag` which is given by the complement of the input `flag`.
    new_flag = 1 - flag

    # Increases the length of `new_flag` by adding a flagged entry at its
    # beginning and another flagged entry at its end. This will circumvent
    # boundary effects when manipulating the flagged entries.
    new_flag = np.concatenate((np.array([1]), new_flag, np.array([1])))

    # Apply `expand_flag` to `new_flag`, but inverting the order of `b` and `e`.
    new_flag = expand_flag(new_flag, interval=(e, b))

    # Crops `new_flag` to its original length by removing its first and last
    # elements which were artificially introduced above.
    new_flag = np.delete(new_flag, [0, len(new_flag) - 1], axis=0)

    # Obtains the complement of new_flag.
    new_flag = 1 - new_flag

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


def get_closest_slice(target_slice, list_slices):
    """ Gets the slice in `list_slices` which is closest to `target_slice`.

        Selects and outputs the slice object listed in `list_slices` which is
        closest to the `target_slice`. This is done by finding the slice in
        `list_slices` whose center is closest to the center of `target_slice`.

        Args:
            target_slice: a single slice object of the form
                'slice(start, stop, None)'.
            list_slices: a list of slice objects of the form
                'slice(start, stop, None)'.

        Returns:
            A single slice object from `list_slices` whose center is closest
            to the center of `target_slice`.

        Example(s):
            >>> target_slice = slice(2, 5, None)
            >>> list_slices = [slice(1, 3, None), slice(7, 10, None)]
            >>> get_closest_slice(target_slice, list_slices)
            >>> slice(1, 3, None)
    """

    # Creates a list composed of the distances between the centers of each
    # slice object in `list_slices` and the center of `target_slice`.
    distances = [
                 np.r_[entry_slice].mean() - np.r_[target_slice].mean()
                 for entry_slice in list_slices
                 ]

    # Finds which index of `distances` labels the smallest distance.
    index = np.argmin(np.abs(distances))

    # Returns the slice object in `list_slices` given by `index`.
    return list_slices[index]


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
        read_switch=True, read_temp=True, verbose=False):
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
        read_temp: a boolean parameter which determines whether or not the =
            temperature data must be read.
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
        ]
    switch_files = [
        'antenna.scio', 'res100.scio', 'res50.scio', 'short.scio', 'noise.scio'
        ]
    temp_files = [
        ('temp_100A_bot_lna.raw', 'float'), ('temp_100_ambient.raw', 'float'),
        ('temp_100A_noise.raw', 'float'), ('temp_100A_switch.raw', 'float'),
        ('temp_100A_top_lna.raw', 'float'), ('temp_100B_bot_lna.raw', 'float'),
        ('temp_100B_noise.raw', 'float'), ('temp_100B_switch.raw', 'float'),
        ('temp_100B_top_lna.raw', 'float'), ('temp_70A_bot_lna.raw', 'float'),
        ('temp_70_ambient.raw', 'float'), ('temp_70A_noise.raw', 'float'),
        ('temp_70A_switch.raw', 'float'), ('temp_70A_top_lna.raw', 'float'),
        ('temp_70B_bot_lna.raw', 'float'), ('temp_70B_noise.raw', 'float'),
        ('temp_70B_switch.raw', 'float'), ('temp_70B_top_lna.raw', 'float'),
        ('temp_pi.raw', 'float'), ('temp_snapbox.raw', 'float'),
        ('time_pi.raw', 'float'), ('time_start_therms.raw', 'float'),
        ('time_stop_therms.raw', 'float'),
        ]
        
    # Lists the old and new time '.raw' file names, their respective data types,
    # and their new file nomemclature.
    old_time_raw_files = [
        ('time_start.raw', 'float', 'time_sys_start.raw'),
        ('time_stop.raw', 'float', 'time_sys_stop.raw'),
        ('time_rtc_start.raw', 'float', 'time_rtc_start.raw'),
        ('time_rtc_stop.raw', 'float', 'time_rtc_stop.raw'),
        ]
    new_time_raw_files = [
            ('time_sys_start.raw', 'float', 'time_sys_start.raw'),
            ('time_sys_stop.raw', 'float', 'time_sys_stop.raw'),
            ('time_rtc_start.raw', 'float', 'time_rtc_start.raw'),
            ('time_rtc_stop.raw', 'float', 'time_rtc_stop.raw'),
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

        # Checks whether `first_ctime` is smaller than 1524730000. If so,
        # attempts to read the timestamp information from those files listed in
        # `old_time_raw_files`. This step is needed because at that ctime in
        # 2018 the timestamp files were renamed from 'time_start.raw' and
        # 'time_stop.raw' to 'time_sys_start.raw' and 'time_sys_stop.raw'. Since
        # the timestamp information is essential for any analysis of the PRIZM
        # data, checking whether these older files are available is essential.
        # Notice that despite having different names, the data dictionary keys
        # referring to such files still reflect the more recent file
        # nomenclature in order to keep the resulting `prizm_data` dictionary
        # compatible with other functions defined in this module.
        if first_ctime < 1524730000:
            # Verbose message.
            if verbose:
                print('Attempting to read the older timestamp files.')

            # Reads all '.raw' files in `dirs` whose names match the entries in
            # `old_time_raw_files`. The results are stored in the appropriate
            # antenna dictionary entry of `prizm_data` under a key given by the
            # more recent file nomenclature associated with those files.
            for old_file_name, dtype, file_name in old_time_raw_files:
                prizm_data[antenna][file_name] = read_raw_file(dirs,
                                                               old_file_name,
                                                               verbose=verbose,
                                                               dtype=dtype)
        else:
            # In case `first_ctime` does not fall within the time period which
            # corresponds to the older timestamp files, that means only newer
            # files will be read. The `prizm_data` timestamp entries are thus
            # initialized with an empty NumPy array. This guarantees the next
            # reading operation below can be performed even if no old timestamp
            # data has been yet recorded in `prizm_data`.
            for old_file_name, dtype, file_name in old_time_raw_files:
                prizm_data[antenna][file_name] = np.array([])

        # Checks whether `second_ctime` is larger than 1524730000. If so,
        # attempts to read the timestamp information from those files listed in
        # `new_time_raw_files`. Here the operation concatenates the output of
        # `read_raw_file` to the `prizm_data` timestamp entries initialized
        # above. This guarantees that the reading operation will work as
        # expected in all scenarios, i.e., when there is only old data, or
        # only new data, or a mix of old and new data.
        if second_ctime > 1524730000:
            # Verbose message.
            if verbose:
                print('Attempting to read the newer timestamp files.')

            # Reads all '.raw' files in `dirs` whose names match the entries in
            # `new_time_raw_files`. The results are temporatily stored in the
            # NumPy array `new_time_data`. It is then concatenated to the
            # timestamp information already stored or initialized in the antenna
            # dictionary entry of `prizm_data`.
            for new_file_name, dtype, file_name in new_time_raw_files:
                new_time_data = read_raw_file(dirs,
                                              new_file_name,
                                              verbose=verbose,
                                              dtype=dtype)
                prizm_data[antenna][file_name] = np.concatenate((
                                                 prizm_data[antenna][file_name],
                                                 new_time_data
                                                 ))

        # Reads all remaining '.raw' files in `dirs` whose names match the
        # entried in `raw_files`. The results are stored in the appropriate
        # atenna dictionary entry of `prizm_data` with key given by the file
        # name.
        for file_name, dtype in raw_files:
            prizm_data[antenna][file_name] = read_raw_file(dirs,
                                                           file_name,
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
            print('Reading the auxiliary switch data.')

        # Reads all '.scio' files in `dirs` whose names match the entries in
        # `switch_files`. The results are stored as dictionaries in
        # `prizm_data['switch']`, with keys given by the file names being read.
        for file_name in switch_files:
            prizm_data['switch'][file_name] = read_scio_file(dirs,
                                                             file_name,
                                                             verbose=verbose)

    # Checks whether `read_temp` is `True`. If so, the key `temp` is added
    # to the `prizm_data` dictionary, creates the list of directories where
    # the temperature data is located, and proceeds to read the data.
    if read_temp:
        prizm_data['temp'] = {}
        dirs = dir_from_ctime(first_ctime,
                              second_ctime,
                              dir_top + '/' + subdir_switch)

        # Verbose message.
        if verbose:
            print('Reading the auxiliary temperature data.')

        # Reads all '.scio' files in `dirs` whose names match the entries in
        # `switch_files`. The results are stored as dictionaries in
        # `prizm_data['switch']`, with keys given by the file names being read.
        for file_name, dtype in temp_files:
            prizm_data['temp'][file_name] = read_raw_file(dirs,
                                                          file_name,
                                                          verbose=verbose,
                                                          dtype=dtype)

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
            # `times`. These are used to create a filter `chunk_filter` which
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


def add_temp_flags(prizm_data, antennas=['70MHz', '100MHz']):
    """ Creates a 'temp_flags' entry in a PRIZM data dictionary.

    Adds a 'temp_flags' entry for each of the `antennas` featuring in the
    input `prizm_data` dictionary. These new entries are based on the auxiliary
    'temp' entry contained in that same dictionary which contains thermometry
    information associated with different PRIZM internal components.

    Args:
        prizm_data: a dictionary containing all PRIZM data structured according
            to the output of the function `read_prizm_data`.
        antennas: a list containing the antennas for flag generation.

    Returns:
        The input dictionary with an additional entry with key 'temp_flags'
        for each antenna listed in `antennas`. The new entry contains a NumPy
        array which indicates when temperature measurements were being
        performed. A typical output returned by this function would have the
        following structure.

        {
        '70MHz': {
            'pol0.scio': numpy.array,
            ...,
            'temp_flags': numpy.array,
            },
        '100MHz': {
            'pol0.scio': numpy.array,
            ...,
            'time_sys_stop.raw',
            'temp_flags': numpy.array,
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
                '`add_temp_flags`: the data for the '
                + antenna
                + ' antenna could not be found.'
                )
            continue

        # Makes sure the input dictionary contains the timestamp data. An error
        # message is printed if that information is missing.
        if len(prizm_data[antenna]['time_sys_start.raw']) == 0:
            print(
                '`add_temp_flags`: no timestamp data was found for the '
                + antenna
                + ' antenna.'
                )
            continue

        # Initializes the dictionary entry which will store the flags.
        prizm_data[antenna]['temp_flags'] = {}

        # Here `therms_time_start` and `therms_time_stop` contain the ctimes at
        # which the data-taking associated with a the instrument's thermometers
        # started and stopped, respectively.
        therms_time_start = prizm_data['temp']['time_start_therms.raw']
        therms_time_stop = prizm_data['temp']['time_stop_therms.raw']

        # Initializes the NumPy array `flag` which will be used in the flags
        # generation below.
        flag = np.zeros_like(prizm_data[antenna]['time_sys_start.raw'],
                             dtype='int')

        # Takes the ctime data stored in `prizm_data` in preparation for the
        # data chunk selection performed below. Here the NumPy arrays
        # `data_time_start` and `data_time_stop` contain the times at which
        # data (associated with any) given component started and stopped
        # being recorded, respectively.
        data_time_start = prizm_data[antenna]['time_sys_start.raw']

        # Slices the data into chunks delimited in time by the entries in
        # `therms_time_start` and `therms_time_stop`. These are used to create a
        # filter `chunk_filter` which picks only data matching the chunk under
        # consideration. Notice that we add a buffer of `4.3` (corresponding
        # to half of the sampling time in `data_time_start`) when generating
        # `chunk_filter` in order to avoid creating single-sample flags.
        for chunk_start, chunk_end in zip(therms_time_start, therms_time_stop):
            condition = np.logical_and(data_time_start >= chunk_start - 4.3,
                                       data_time_start <= chunk_end + 4.3)
            chunk_filter = np.where(condition)[0]

            # Assigns the value `1` to the portions of `flag` corresponding to
            # the chunk under consideration.
            flag[chunk_filter] = np.ones(len(chunk_filter), dtype='int')

        # Adds flags to `prizm_data`.
        prizm_data[antenna]['temp_flags'] = flag

    return


def add_nighttime_flags(prizm_data, antennas=['70MHz', '100MHz']):
    """ Creates a 'nighttime_flags' entry in a PRIZM data dictionary.
    
    Adds a 'nighttime_flags' entry for each of the `antennas` featuring in the
    input `prizm_data` dictionary. These new entries are based on the sunset
    time at the experiment's location and flags the data which was acquired by
    the instrument during nighttime.

    Args:
        prizm_data: a dictionary containing all PRIZM data structured according
            to the output of the function `read_prizm_data`.
        antennas: a list containing the antennas for flag generation.

    Returns:
        The input dictionary with an additional entry with key 'nighttime_flags'
        for each antenna listed in `antennas`. The new entry contains a NumPy
        array which indicates which portions of the data were recorded during
        nighttime. A typical output returned by this function would have the
        following structure.

        {
        '70MHz': {
            'pol0.scio': numpy.array,
            ...,
            'nighttime_flags': numpy.array,
            },
        '100MHz': {
            'pol0.scio': numpy.array,
            ...,
            'time_sys_stop.raw',
            'nighttime_flags': numpy.array,
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
                '`add_nighttime_flags`: the data for the '
                + antenna
                + ' antenna could not be found.'
                )
            continue

        # Makes sure the input dictionary contains the timestamp data. An error
        # message is printed if that information is missing.
        if len(prizm_data[antenna]['time_sys_start.raw']) == 0:
            print(
                '`add_nighttime_flags`: no timestamp data was found for the '
                + antenna
                + ' antenna.'
                )
            continue

        # Extracts the NumPy array of data ctimes stored in `prizm_data`.
        ctimes = prizm_data[antenna]['time_sys_start.raw']

        # Obtains the timestamps associated with these `ctimes`.
        dates = timestamp_from_ctime(ctimes)

        # Transforms the entries of `dates` into datetime objects.
        dates = [
            datetime.strptime(entry, '%Y%m%d_%H%M%S')
            for entry in dates
            ]

        # Artificially includes a one-day buffer to `dates` in order to
        # facilitate the flagging of nighttime data below.
        previous_to_first_day = dates[0] - timedelta(days=1)
        next_to_last_day = dates[-1] + timedelta(days=1)
        dates = [previous_to_first_day] + dates + [next_to_last_day]

        # Initializes the geographial location of Marion island for the purpose
        # of obtaining accurate sunset and sunrise times for different periods
        # of the year.
        marion = suntime.Sun(lat=-46.88694, lon=37.819638)

        # Obtains the sunrise and sunset times for the time period spanned by
        # `dates`.
        sunset_times = [
            marion.get_sunset_time(entry)
            for entry in dates
            ]

        sunrise_times = [
            marion.get_sunrise_time(entry)
            for entry in dates
            ]

        # Keeps only the unique entries featuring in `sunset_times` and
        # `sunrise_times`.
        sunset_times = np.unique(sunset_times)
        sunrise_times = np.unique(sunrise_times)

        # Sets the reference time `ref_time` used in the definition of ctime
        # (i.e., the number of seconds since 1970/1/1).
        ref_time = datetime(1970, 1, 1, tzinfo=timezone.utc)

        # Transforms the datetime objects in `sunset_times` to ctimes. As a
        # buffer,
        # we also add an hour (3600 seconds) to the sunset ctimes and subtract
        # the
        # same amount of seconds from the sunrise ctimes.
        sunset_ctimes = [
                        (entry - ref_time).total_seconds() + 3600
                        for entry in sunset_times
                        ]

        sunrise_ctimes = [
                        (entry - ref_time).total_seconds() - 3600
                        for entry in sunrise_times
                        ]

        # Deletes the first entry of `sunrise_ctimes` and the last entry of
        # `sunset_ctimes`. This creates an offset between the two NumPy arrays
        # which makes the process of finding the nighttime portions of the data a
        # lot simpler.
        sunrise_ctimes = sunrise_ctimes[1:]
        sunset_ctimes = sunset_ctimes[:-1]

        # Initializes the NumPy array `flag` which will be used in the flags
        # generation below.
        flag = np.zeros_like(prizm_data[antenna]['time_sys_start.raw'],
                            dtype='int')

        # Slices the data into chunks delimited in time by the entries in
        # `sunset_ctimes` and `sunrise_ctimes`. These are used to create a
        # `nighttime_filter` which picks only nighttime data.
        for nighttime_start, nighttime_end in zip(sunset_ctimes, sunrise_ctimes):
            condition = np.logical_and(ctimes >= nighttime_start,
                                       ctimes <= nighttime_end)
            nighttime_filter = np.where(condition)[0]

            # Assigns the value `1` to the portions of `flag` corresponding to
            # the chunk under consideration.
            flag[nighttime_filter] = np.ones(len(nighttime_filter), dtype='int')

        # Adds flags to `prizm_data`.
        prizm_data[antenna]['nighttime_flags'] = flag

    return


def add_moon_flags(prizm_data, antennas=['70MHz', '100MHz'], altitude_buffer=10):
    """ Creates a 'moon_flags' entry in a PRIZM data dictionary.
    
    Adds a 'moon_flags' entry for each of the `antennas` featuring in the input
    `prizm_data` dictionary. These new entries are based on the Moon's altitude
    at the experiment's location and flags the data which was acquired by the
    instrument as being acquired when the Moon was above the horizon.

    Args:
        prizm_data: a dictionary containing all PRIZM data structured according
            to the output of the function `read_prizm_data`.
        antennas: a list containing the antennas for flag generation.

    Returns:
        The input dictionary with an additional entry with key 'moon_flags' for
        each antenna listed in `antennas`. The new entry contains a NumPy array
        which indicates which portions of the data were recorded when the Moon
        was above the horizon. A typical output returned by this function would
        have the following structure.

        {
        '70MHz': {
            'pol0.scio': numpy.array,
            ...,
            'moon_flags': numpy.array,
            },
        '100MHz': {
            'pol0.scio': numpy.array,
            ...,
            'time_sys_stop.raw',
            'moon_flags': numpy.array,
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
                '`add_moon_flags`: the data for the '
                + antenna
                + ' antenna could not be found.'
                )
            continue

        # Makes sure the input dictionary contains the timestamp data. An error
        # message is printed if that information is missing.
        if len(prizm_data[antenna]['time_sys_start.raw']) == 0:
            print(
                '`add_moon_flags`: no timestamp data was found for the '
                + antenna
                + ' antenna.'
                )
            continue

        # Extracts the NumPy array of data ctimes stored in `prizm_data`.
        ctimes = prizm_data[antenna]['time_sys_start.raw']

        # Transforms the entries of `ctimes` into datetime objects.
        dates = [
            datetime.utcfromtimestamp(entry)
            for entry in ctimes
            ]

        # Initializes the geographial location of Marion island for the purpose
        # of obtaining accurate Moon altitutdes for different `dates`.
        marion = ephem.Observer()
        marion.lat = -46.88694
        marion.lon = 37.819638

        # Initializes the NumPy array `flags` which will be used to store `1`
        # if the Moon's altitude is positive and larger than `abs(altitude_buffer)`,
        # `-1` if the Moon's altitude is negative and smaller than
        # `-abs(altitude_buffer)`, and `0` otherwise.
        flags = np.zeros_like(prizm_data[antenna]['time_sys_start.raw'],
                             dtype='int')

        # Extracts the Moon altitude for each entry in `dates` and flags it
        # accordingly.
        for entry, date in enumerate(dates):
            # Sets the current date.
            marion.date = date

            # Extracts the Moon's altitude.
            moon = ephem.Moon(marion)
            altitude = moon.alt

            # Adds an entry to `flags`.
            if altitude > abs(altitude_buffer):
                flags[entry] = 1
            elif altitude < -abs(altitude_buffer):
                flags[entry] = -1

        # Adds flags to `prizm_data`.
        prizm_data[antenna]['moon_flags'] = flags

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
                  '`add_quality_flags`: no timestamp data was found for the '
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


def get_temp_from_slice(prizm_data, antenna, target_slice):
    """
    """

    # Gathers the temperature flags generated by `add_temp_flags`.
    temp_flags = prizm_data[antenna]['temp_flags']

    # Converts the above flag array into NumPy masked arrays.
    temp_flags = np.ma.array(temp_flags, mask=temp_flags)

    # Obtains the a list of slices which group the different flags into
    # contiguous clumps.
    temp_clumps = np.ma.clump_masked(temp_flags)

    # Finds the slice in the `list_slices` of temperature measurements which is
    # closest to the input `target_slice`.
    temp_slice = get_closest_slice(target_slice, temp_clumps)

    # Uses `temp_slice` to determine which ctimes in `prizm_data` correspond to
    # the temperature measurement of interest.
    ctimes = prizm_data[antenna]['time_sys_start.raw'][temp_slice]

    # Loads the thermometer measurement times.
    therms_time_start = prizm_data['temp']['time_start_therms.raw']

    # Determines the `index` which is satisfied by those entries of
    # `therms_time_start` corresponding to those `ctimes` selected above.
    index = np.where(therms_time_start >= ctimes[0])[0][0]

    # Selects the appropriate temperature '.raw' file.
    temp_raw_file = 'temp_' + antenna[:-3] + '_ambient.raw'

    # Uses `index` to extract the ambient temperatures of interest, and store
    # its value in `temp_amb`.
    temp_amb = prizm_data['temp'][temp_raw_file][index]

    # Returns the desired temperatures.
    return temp_amb


def get_spectrum_from_slice(channel_spectra, slice):
    """
    """

    # Finds the middle index associated with of the input `slice`.
    index = int(np.r_[slice].mean())

    # Extracts the spectrum associated with the above `index`.
    spectrum = channel_spectra[index]

    # Returns the obtained spectrum.
    return spectrum


def add_single_gain(prizm_data, antenna, switch_file, clump_nmbr, short_slice, res_slice):
    """
    """

    # Sets some shorter names for convenience.
    gains_pol0 = prizm_data[antenna]['gain']['pol0.scio']
    gains_pol1 = prizm_data[antenna]['gain']['pol1.scio']

    # Checks whether the input `short_slice` and `res_slice` are empty. If so,
    # the gain correction is added to `prizm_data` as an array of zeroes.
    if short_slice == slice(0, 0, None) and res_slice == slice(0, 0, None):
        # Sets the gain the zero.
        zero_gain = np.zeroes(prizm_data[antenna]['pol0.scio'].shape[1])

        # Adds the above gain entry to `prizm_data`.
        gains_pol0[switch_file] = np.stack(gains_pol0[switch_file], zero_gain)
        gains_pol1[switch_file] = np.stack(gains_pol1[switch_file], zero_gain)

    # Else, compute the gain as usual for each polarization channel and store
    # that information in `prizm_data`.
    else:
        # Find the ambient temperature measurement which was performed
        # around the time the instrument observation mode corresponded
        # to `short.scio`.
        temp = get_temp_from_slice(prizm_data, antenna, short_slice)

        # Gets the central cross-sections of the `short.scio` spectra for both
        # polarization channels.
        short_pol0 = get_spectrum_from_slice(prizm_data[antenna]['pol0.scio'],
                                            short_slice
                                            )

        short_pol1 = get_spectrum_from_slice(prizm_data[antenna]['pol1.scio'],
                                            short_slice
                                            )

        # Gets the central cross-sections of either the `res50.scio` or
        # `res100.scio`spectra for both polarization channels.
        res_pol0 = get_spectrum_from_slice(prizm_data[antenna]['pol0.scio'],
                                           res_slice
                                           )

        res_pol1 = get_spectrum_from_slice(prizm_data[antenna]['pol1.scio'],
                                           res_slice
                                           )

        # Computes the gain corrections for each polarization.
        pol0_gain = temp/(res_pol0 - short_pol0)
        pol1_gain = temp/(res_pol1 - short_pol1)

        # Stores the resulting gain correction arrays under the appropriate
        # entries of the `prizm_data` data dictionary.
        gains_pol0[switch_file][clump_nmbr,:] = pol0_gain
        gains_pol1[switch_file][clump_nmbr,:] = pol1_gain

    return


def add_multiple_gains(prizm_data, antennas=['70MHz', '100MHz']):
    """ Short documentation string.

    Long documentation string.

    Args:
        prizm_data: a dictionary containing all PRIZM data structured according
            to the output of the function `read_prizm_data`.
        antennas: a list containing the antennas for flag generation.

    Returns:
        Return object documentation string.
    """

    # Get the gain corrections for each antenna.
    for antenna in antennas:

        # Gathers the switch flags generated by `add_switch_flags`.
        antenna_flags = prizm_data[antenna]['switch_flags']['antenna.scio']
        short_flags = prizm_data[antenna]['switch_flags']['short.scio']
        res50_flags = prizm_data[antenna]['switch_flags']['res50.scio']
        res100_flags = prizm_data[antenna]['switch_flags']['res100.scio']
        noise_flags = prizm_data[antenna]['switch_flags']['noise.scio']

        # Computes the total number of samples associated with the
        # calibration data.
        n_samples = (short_flags.sum()
                     + res50_flags.sum()
                     + res100_flags()
                     + noise_flags())

        # Trims the first and final entries of the switch flags gathered above
        # in order to avoid leakage of information from one observing mode to
        # another.
        antenna_flags = shrink_flag(antenna_flags, (1,1))
        short_flags = shrink_flag(short_flags, (1,1))
        res50_flags = shrink_flag(res50_flags, (1,1))
        res100_flags = shrink_flag(res100_flags, (1,1))
        res100_flags = shrink_flag(noise_flags, (1,1))

        # Converts the above flag arrays into NumPy masked arrays. These have
        # many useful properties which make it much easier to split flags into
        # contiguous clumps.
        antenna_flags = np.ma.array(antenna_flags, mask=antenna_flags)
        short_flags = np.ma.array(short_flags, mask=short_flags)
        res50_flags = np.ma.array(res50_flags, mask=res50_flags)
        res100_flags = np.ma.array(res100_flags, mask=res100_flags)
        noise_flags = np.ma.array(noise_flags, mask=noise_flags)

        # Obtains the a NumPy array of slices which group the different flags
        # into contiguous clumps.
        antenna_clumps = np.array(np.ma.clump_masked(antenna_flags))
        short_clumps = np.array(np.ma.clump_masked(short_flags))
        res50_clumps = np.array(np.ma.clump_masked(res50_flags))
        res100_clumps = np.array(np.ma.clump_masked(res100_flags))
        noise_clumps = np.array(np.ma.clump_masked(noise_flags))

        # Determines the number of clumps in the data, as well as the
        # number of frequencies is spans.
        n_clumps = len(antenna_clumps)
        n_freq = prizm_data[antenna]['pol0.scio'].shape[1]

        # Initializes the `prizm_data` entries which will hold the gain
        # corrections.
        prizm_data[antenna]['gain'] = {
            'pol0.scio': {'res50.scio': np.zeros((n_clumps, n_freq)),
                          'res100.scio': np.zeros((n_clumps, n_freq)),
                          'noise.scio': np.zeros((n_clumps, n_freq)),
                          },
            'pol1.scio':  {'res50.scio': np.zeros((n_clumps, n_freq)),
                           'res100.scio': np.zeros((n_clumps, n_freq)),
                           'noise.scio': np.zeros((n_clumps, n_freq)),
                           },
            }

        # Computes the approximate average number of time samples which forms
        # the gap between two entries of `antenna_clumps`.
        antenna_gap = int(n_samples/n_clumps)

        # Creates auxiliary arrays to facilitate the comparison between the
        # clumps associated with the different observing modes. These are
        # essentially the shifted versions of `antenna_clumps` with either its
        # last entry deleted and an artificial initial entry given by
        # `start_clump`, or first entry deleted and an artificial last entry
        # given by `end_clump`.
        start_clump = slice(antenna_clumps[0].start - antenna_gap,
                            antenna_clumps[0].stop - antenna_gap + 1,
                            None)
        previous_clumps = np.delete(antenna_clumps, -1)
        previous_clumps = np.insert(previous_clumps, 0, start_clump)

        end_clump = slice(antenna_clumps[-1].start + antenna_gap,
                          antenna_clumps[-1].stop + antenna_gap + 1,
                          None)
        next_clumps = np.delete(antenna_clumps, 0)
        next_clumps = np.insert(next_clumps, -1, end_clump)

        # Zip the different versions of `antenna_clumps` generated above into an
        # iterator which will help us make useful comparisons below.
        clumps = zip(previous_clumps, antenna_clumps, next_clumps)

        # Loops over all `antenna_clumps`, and computes the gain correction for
        # the portion of the data to which they correspond. The result for each
        # clump is stored as a NumPy array in the `prizm_data` dictionary.
        for i, (previous_clump, current_clump, next_clump) in enumerate(clumps):
            # Checks whether the `current_clump` has accompanying `short.scio`,
            # `res50.scio`, `short.scio`, `res100.scio`, and `noise.scio` clumps
            # occurring either all before or both after it.
            
            # Determines which clumps occur immediatelly after.
            short_after = np.logical_and(short_clumps > current_clump,
                                         short_clumps < next_clump)
            res50_after = np.logical_and(res50_clumps > current_clump,
                                         res50_clumps < next_clump)
            res100_after = np.logical_and(res100_clumps > current_clump,
                                          res100_clumps < next_clump)

            # Obtains the truth value of the intersection of `short_after` with
            # `res50_after` and `res100_after`.
            short_res50_after = np.logical_and(short_after.any() == True,
                                               res50_after.any() == True)
            short_res100_after = np.logical_and(short_after.any() == True,
                                                res100_after.any() == True)

            # Filters those clumps which occur immediatelly before.
            short_before = np.logical_and(short_clumps > previous_clump,
                                          short_clumps < current_clump)
            res50_before = np.logical_and(res50_clumps > previous_clump,
                                          res50_clumps < current_clump)
            res100_before = np.logical_and(res100_clumps > previous_clump,
                                           res100_clumps < current_clump)

            # Obtains the truth value of the intersection of `short_before` with
            # `res50_before` and `res100_before`.
            short_res50_before = np.logical_and(short_before.any() == True,
                                                res50_before.any() == True)
            short_res100_before = np.logical_and(short_before.any() == True,
                                                res100_before.any() == True)

            # Computes the gain correction using the those clumps associated
            # with `res50.scio` and `short.scio`. It always attempts to use
            # those auxiliary clumps that come after the `current_clump`. When
            # those are incomplete or not present, it attempts to use those
            # auxiliary clumps that come before the `current_clump`. If these
            # are also incomplete or not present, the gain correction is then
            # set to an array of zeros.
            if short_res50_after == True:
                # Picks the relevant clump associated with `short.scio` and
                # `res50.scio`.
                short_clump = short_clumps[short_after][0]
                res50_clump = res50_clumps[res50_after][0]

                # Adds the gain correction for the `current_clump`.
                add_single_gain(prizm_data,
                                antenna,
                                'res50.scio',
                                i,
                                short_clump,
                                res50_clump)

            elif short_res50_before == True:
                # Picks the relevant clump associated with `short.scio` and
                # `res50.scio`.
                short_clump = short_clumps[short_before][0]
                res50_clump = res50_clumps[res50_before][0]

                # Adds the gain correction for the `current_clump`.
                add_single_gain(prizm_data,
                                antenna,
                                'res50.scio',
                                i,
                                short_clump,
                                res50_clump)

            else:
                # Sets the clumps associated with `short.scio` and `res50.scio`
                # as empty slices.
                short_clump = slice(0, 0, None)
                res50_clump = slice(0, 0, None)
            
                # Set gain corrections to NumPy arrays of zeros.
                add_single_gain(prizm_data,
                                antenna,
                                'res50.scio',
                                i,
                                short_clump,
                                res50_clump)

            # Computes the gain correction using the those clumps associated
            # with `res100.scio` and `short.scio`. It always attempts to use
            # those auxiliary clumps that come after the `current_clump`. When
            # those are incomplete or not present, it attempts to use those
            # auxiliary clumps that come before the `current_clump`. If these
            # are also incomplete or not present, the gain correction is then
            # set to an array of zeros.
            if short_res100_after == True:
                # Picks the relevant clump associated with `short.scio` and
                # `res100.scio`.
                short_clump = short_clumps[short_after][0]
                res100_clump = res100_clumps[res100_after][0]

                # Adds the gain correction for the `current_clump`.
                add_single_gain(prizm_data,
                                antenna,
                                'res100.scio',
                                i,
                                short_clump,
                                res100_clump)

            elif short_res100_before == True:
                # Picks the relevant clump associated with `short.scio` and
                # `res100.scio`.
                short_clump = short_clumps[short_before][0]
                res100_clump = res100_clumps[res100_before][0]

                # Adds the gain correction for the `current_clump`.
                add_single_gain(prizm_data,
                                antenna,
                                'res100.scio',
                                i,
                                short_clump,
                                res100_clump)

            else:
                # Sets the clumps associated with `short.scio` and `res50.scio`
                # as empty slices.
                short_clump = slice(0, 0, None)
                res100_clump = slice(0, 0, None)

                # Set gain corrections to NumPy arrays of zeros.
                add_single_gain(prizm_data,
                                antenna,
                                i,
                                'res50.scio',
                                short_clump,
                                res100_clump)

    return


def read_beam(dir_parent, file_name):
    """ Reads a beam simulation file for a given PRIZM antenna.

    Looks for files with the given `file_name` under the input parent directory
    `dir_parent`. If the file has been located in the provided directory, the
    function attempts to read it. In case the file cannot be found and/or read,
    an error message is printed. Successfully read files are stored and returned
    in a dictionary format.

    Args:
        dir_parent = a string containing the top level directory where the file
            is stored.
        file_name: a string specifying the beam simulation file for the antenna
            of interest.

    Returns:
        A dictionary containing the sampled beam amplitudes and associated
        spherical coordinates for each frequency channel (in MHz). A typical
        output returned by this function would have the following structure.

        {
        'theta': numpy.array,
        'phi': numpy.array,
        20: numpy.array,
        22: numpy.array,
        ...,
        198: numpy.array,
        200: numpy.array,
        }
    """

    # Initializes the dictionary which will hold the beam information.
    beam_dict = {}

    # Establishes the `file_path` which points to the beam simulation of
    # interest.
    file_path = dir_parent + '/' + file_name

    # Stores the beam simulation data in the NumPy array `beam_sim_data`, and
    # ignores the header as a comment starting with '#'.
    beam_sim_data = np.loadtxt(file_path, delimiter=',', comments='#')

    # Reads the beam file header, cleans it from unwanted characters, and keeps
    # only the numerical entries – these correspond to the different frequencies
    # for which the beam has been simulated.
    beam_file = open(file_path, 'r')
    header = beam_file.readline()
    frequencies = header.strip('#\n, ').split(',')[2:]
    beam_file.close()

    # Converts the `frequencies` list to a NumPy array and converts its values
    # to MHz through a division by 1e6.
    frequencies = np.asarray(frequencies, dtype='float')/1e6

    # Extracts the spherial coordinates `theta` and `phi` stored in
    # `beam_sim_data` and converts their units from degrees to radians.
    theta = np.unique(beam_sim_data[:,0])*np.pi/180
    phi = np.unique(beam_sim_data[:,1])*np.pi/180

    # Discards the coordinate information from `beam_sim_data` since this is
    # already stored in the meshgrid, as well as in `theta` and `phi`.
    beam_sim_data = beam_sim_data[:,2:]

    # Stores spherical coordinates in `beam_dict`.
    beam_dict['theta'] = theta
    beam_dict['phi'] = phi
    
    # Stores the beam profile for each frequency in `beam_dict`.
    for index, entry in enumerate(frequencies):
        # Reshape the `beam_sim_data` so that its dimensions are compatible with
        # those of `theta` and `phi`. This way different slices of `beam_sim_data`
        # correspond to the beam for different frequencies.
        reshaped_beam_sim = np.reshape(beam_sim_data[:,index],
                                       [len(phi), len(theta)])
        
        # Stores the reshaped beam in `beam_dict` under the appropriate
        # frequency key.
        beam_dict[entry] = reshaped_beam_sim

    # Returns the beam information in a dictionary format.
    return beam_dict


def healpy_beam(beam_dict, healpy_nside=256, site_latitude=-46.88694):
    """ Converts a beam simulation dictionary into HealPix format.

    Given an input dictionary `beam_dict` containing the raw beam simulation and
    associated spherical coordinates, generates a new dictionary is generated in
    which the beam amplitudes and spherical coordinates are adapted to the
    HealPix format with pixelization set by `healpy_nside`.

    Args:
        beam_dict: a dictionary containing a raw beam simulation.
        healpy_nside: an integer specipying the HealPix pixelization.
        site_latitude: the latitute of the instrument associated with the beam.

    Returns:
        A dictionary containing the sampled HealPix beam amplitudes and
        associated spherical coordinates for each frequency channel (in MHz). A
        typical output returned by this function would have the following
        structure.

        {
        'theta': numpy.array,
        'phi': numpy.array,
        20: numpy.array,
        22: numpy.array,
        ...,
        198: numpy.array,
        200: numpy.array,
        'normalization': numpy.array,
        }
    """

    # Initializes the dictionary which will hold the HealPy version of the beam.
    healpy_beam_dict = {}

    # Extracts the frequencies for which beams are available in `beam_dict`.
    frequencies = [key for key in beam_dict.keys() if isinstance(key, float)]
    n_freq = len(frequencies)
    
    # Initializes a HealPy pixelization and associated spherical coordinates.
    healpy_npix = healpy.nside2npix(healpy_nside)
    healpy_theta, healpy_phi = healpy.pix2ang(healpy_nside,
                                              np.arange(healpy_npix))

    # Stores spherical coordinates in `healpy_beam_dict`.
    healpy_beam_dict['theta'] = healpy_theta
    healpy_beam_dict['phi'] = healpy_phi

    # SciPy 2D interpolation forces us to do proceed in chunks of constant
    # coordinate `healpy_theta`. Below we find the indices at which
    # `healpy_theta` changes.
    indices = np.where(np.diff(healpy_theta) != 0)[0]
    indices = np.append(0, indices + 1)

    # Initializes the NumPy array which will contain the normalization factor
    # for each beam.
    beam_norms = np.zeros(n_freq)

    # Loops over the different frequencies for which the beam has been
    # simulated.
    for i, frequency in enumerate(frequencies):

        # Computes the actual beam from the information contained in
        # `beam_dict`.
        beam = 10**(beam_dict[frequency]/10)

        # Interpolates beam.
        beam_interp = interpolate.interp2d(beam_dict['theta'],
                                           beam_dict['phi'],
                                           beam,
                                           kind='cubic',
                                           fill_value=0)

        # Initializes `healpy_beam`, the HealPy version of the beam.
        healpy_beam = np.zeros(len(healpy_theta))

        # Constructs the HealPy beam.
        for j in range(np.int(len(indices)/2) + 2):
            start = indices[j]
            end = indices[j+1]
            healpy_beam[start:end] = beam_interp(healpy_theta[start],
                                             healpy_phi[start:end])[:,0]

        # Fills `beam_norms` with the appropriate normalization factors for
        # each HealPy beam.
        beam_norms[i] = np.sqrt(np.sum(healpy_beam**2))
        
        # Rotates and stores the the HealPy beam in the `healpy_beam_dict` under
        # the appropriate frequency entry.
        beam_rotation = healpy.rotator.Rotator([0, 0, 90 - site_latitude])
        healpy_beam = beam_rotation.rotate_map_pixel(healpy_beam/beam_norms[i])
        healpy_beam_dict[frequency] = healpy_beam

    # Adds the beam normalizations as a separate entry in `heapy_beam_dict`.
    healpy_beam_dict['normalization'] = beam_norms

    # Returns the HealPy version of the beam in a dictionary format.
    return healpy_beam_dict


def beam_covariance(beam_dict):
    """
    """

    # Extracts the beam profiles from the input beam dictionary `beam_dict`.
    beam_array = np.array([
                           entry.tolist()
                           for key, entry in beam_dict.items()
                           if isinstance(key, float)
                           ])
    
    # Computes the beam covariance.
    beam_covariance = np.dot(beam_array, beam_array.T)

    # Returns the `beam_covariance`.
    return beam_covariance


# Jon's harmonic space product.
def almsdotalms(alms1,alms2,nside=None):
    nalm=len(alms2)
    lmax=np.int(np.sqrt(2*nalm))
    if nside is None:
        nside=np.int(lmax/3)
    if len(alms1.shape)==1:
        ans=2*np.dot(np.conj(alms1[lmax:]),alms2[lmax:])
        ans=ans+np.dot(alms1[:lmax],alms2[:lmax])
    else:
        ans=2*np.dot(np.conj(alms1[lmax:,:].T),alms2[lmax:])
        ans=ans+np.dot(alms1[:lmax,:].T,alms2[:lmax])
    return np.real(ans)*healpy.nside2npix(nside)/4/np.pi
    

