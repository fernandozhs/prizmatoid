# NumPy
import numpy as np

# Copy
import copy

# Scio (https://github.com/sievers/scio)
import scio

# Date and Time
import time


def load_multiple_data(ctime_intervals, components):
    """ Loads PRIZM data associated with a given instrument component and ctime interval. """

    # Initializes the dictionary which will hold the data of interest.
    data = {
            component: {}
            for component in components
            }

    return data


def load_data(metadata, verbose=True):
    """ Loads and patches PRIZM data associated with a given metadatabase entry. """

    # Extracts the component associated with the input `metadata`.
    component = metadata['component']

    # Extracts the `data_subdirectory` and `patches_subdirectory` stored in the input `metadata`.
    data_subdirectories = [
                           data_directory + metadata['data_subdirectory'] + '/' + str(ctime)
                           for ctime in  metadata['ctimes']
                           ]
    patches_subdirectory = patches_directory + metadata['patches_subdirectory']

    # Initializes the dictionary which will hold the data of interest.
    data = {
            component: {}
            }

    # Reads all '.scio' files listed in the input `metadata`.
    for file_name, key in metadata['scio_files']:
        data[component][key] = read_scio_file(
                                              data_subdirectories,
                                              file_name,
                                              verbose
                                              )

    # Reads all '.raw' files listed in the input `metadata`.
    for file_name, dtype, key in metadata['raw_files']:
        data[component][key] = read_raw_file(
                                             data_subdirectories,
                                             file_name,
                                             dtype,
                                             verbose,
                                             )

    # Loads and applies all patch files listed in the input `metadata`.
    #for file_name in metadata['patch_files']:
    #    data[component] = apply_patch(
    #                                  data[component],
    #                                  patches_subdirectory,
    #                                  file_name,
    #                                  verbose
    #                                  )

    return data


def apply_patch(data, dirs, file_name, verbose):
    """ Patches PRIZM data. """

    # Patches the data.
    if file_name == 'rearranging.npy':
        # Loads the input `patch_file`.
        rearranging = np.load(patches_directory + '/' + file_name, allow_pickle=True)

        # Rearranges the information contained in the input `data`.
        data['pol0.scio'] = data['pol0.scio'][rearranging]
        data['pol1.scio'] = data['pol1.scio'][rearranging]
        data['time_sys_start.raw'] = data['time_sys_start.raw'][rearranging]
        data['time_sys_stop.raw'] = data['time_sys_stop.raw'][rearranging]

    # Applies offsets to the `data` timestamps.
    if file_name == 'offsetting.npy':
        # Loads the input `patch_file`.
        offsetting = np.load(patches_directory + '/' + file_name, allow_pickle=True)

        # Offsets the input `data` timestamps.
        for slice, offset in offsetting:
            data['time_sys_start.raw'][slice] += offset
            data['time_sys_stop.raw'][slice] += offset

    # Trims the input `data`.
    if file_name == 'trimming.npy':
        # Loads the patch instructions.
        trimming = np.load(patches_directory + '/' + file_name, allow_pickle=True)

        # Trims the input `data`.
        data['pol0.scio'] = np.delete(data['pol0.scio'], np.r_[tuple(trimming)], axis=0)
        data['pol1.scio'] = np.delete(data['pol1.scio'], np.r_[tuple(trimming)], axis=0)
        data['time_sys_start.raw'] = np.delete(data['time_sys_start.raw'], np.r_[tuple(trimming)], axis=0)
        data['time_sys_stop.raw'] = np.delete(data['time_sys_stop.raw'], np.r_[tuple(trimming)], axis=0)

    return data


def read_scio_file(dirs, file_name, verbose):
    """ Reads '.scio' files located in a given list of directories.

    Looks for files with the given `file_name` in the input list of directories
    `dirs`. If the file has been located in the provided directory, the function
    attempts to read it. In case the file cannot be found and/or read, an error
    message is printed. All files which have been successfully located and read
    are stacked and returned as a single NumPy array.
    (This function is largely equivalent to `prizmtools.read_pol_fast`).

    Args:
        dirs: a list of strings specifying the directories where the '.scio'
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


def read_raw_file(dirs, file_name, dtype, verbose):
    """ Reads '.raw' files located in a given list of directories.

    Looks for files with the given `file_name` in the input list of directories
    `dirs`. If the file has been located in the provided directory, the function
    attempts to read it. In case the file cannot be found and/or read, an error
    message is printed. All files which have been successfully located and read
    are stacked and returned a single NumPy array.
    (This function is largely equivalent to `prizmtools.read_field_many_fast`).

    Args:
        dirs: a list of strings specifying the directories where the '.raw' files
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


def retrieve_multiple_metadata(ctime_intervals, components):
    """ Retrieves the PRIZM metadata associated with multiple instrument components and ctime intervals. """

    # Initializes the list which will hold the `metadata` entries of interest.
    retrieved_metadata = []

    # For each input `component` ...
    for component in components:

        # And each `initial_ctime` and `final_ctime` in the input `ctime_intervals` ...
        for initial_ctime, final_ctime in ctime_intervals:

            # Retrieves the `metadata_entries` of interest.
            metadata_entries = retrieve_metadata(initial_ctime, final_ctime, component)

            # Copies the entries obtained above.
            retrieved_metadata += copy.deepcopy(metadata_entries)

    # Returns a list containing the desired `metadata` entries.
    return retrieved_metadata


def retrieve_metadata(initial_ctime, final_ctime, component):
    """ Retrieves the PRIZM metadata associated with a given instrument component and ctime interval. """

    # Checks whether `initial_ctime` < `final_ctime`. If not, the values of these inputs are swapped.
    if initial_ctime > final_ctime:
        initial_ctime, final_ctime = final_ctime, initial_ctime

    # Extracts the `metadatabase` keys associated with the input ctime range.
    initial_key = int(str(initial_ctime)[:5])
    final_key = int(str(final_ctime)[:5])

    # Collects all `metadatabase` entries associated with the input ctime range.
    retrieved_metadata = [
                          metadata
                          for key in metadatabase[component].keys()
                          for metadata in metadatabase[component][key]
                          if key >= initial_key and key < final_key
                          ]

    # Returns a list containing the desired `metadata` entries.
    return retrieved_metadata


# User-defined directories.
data_directory = '/Volumes/PRIZM_DISK1'
patches_directory = '/Users/Fernando/Desktop/patches_data'


# Metadatabase.
metadatabase = {
    '100MHz':
    {
        15261:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15261',
             'patches_subdirectory': '/marion2018/patches_100MHz/15261/1526100607',
             'ctimes':[1526100607,1526101517,1526102419,1526103330,1526104231,1526105133,1526106045,1526106954,1526107856,1526108763,1526109670,1526110040,1526110103,1526110153,1526110580,1526110817,1526111487,1526111722,1526112388,1526112632,1526113294,1526113539,1526114204,1526114446,1526115106,1526115313,1526115737,1526116180,1526116243,1526116539,1526116598,1526116647,1526116955,1526117278,1526117518,1526118187,1526118423,1526118453,1526119094,1526119742,1526120186,1526121092,1526121993,1526122900,1526123697,1526124602,1526125513,1526125724,1526126625,1526126834,1526127230,1526127293,1526127343,1526127393,1526127888,1526128678,1526128909,1526129642,1526130543,1526131091,1526131423,1526132332,1526132388,1526132438,1526132765,1526132793,1526133717,1526134130,1526135031,1526135933,1526136843,1526136873,1526137774,1526138684,1526139596,1526140501,1526141412,1526142313,1526143216,1526144127,1526145032,1526145299,1526146153,1526147136,1526147515,1526148282,1526149002,1526149204,1526149493,1526149820,1526150527,1526151277,1526151449,1526152348,1526152644,1526153545,1526154452,1526155353,1526156259,1526156881,1526157170,1526157263,1526157668,1526158573,1526159066,1526159717,1526159884,1526160305,1526161215,1526161512,1526162418,1526163298,1526164052,1526164957,1526165859,1526166767,1526167677,1526168587,1526169493,1526169618,1526177060,1526177960,1526178872,1526179779,1526180681,1526181587,1526182488,1526183391,1526184292,1526185199,1526185716,1526186617,1526187523,1526188434,1526189339,1526190247,1526191148,1526191955,1526192055,1526192632,1526192964,1526193015,1526193078,1526193224,1526193540,1526193876,1526194449,1526194704,1526194782,1526195589,1526195687,1526196473,1526196594,1526197374,1526197483,1526198281,1526198394,1526198906,1526199182],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.p','offsetting.p'],
             'channel_0': True,
             'channel_1': True,
             'temperature': True
            },
        ],
    },

    '70MHz':
    {
        15246:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15246',
             'patches_subdirectory': '/marion2018/patches_70MHz/15246/1524600732',
             'ctimes':[1524600732,1524601634,1524602536,1524603438,1524604341,1524605242,1524606144,1524607046,1524607948,1524608850,1524609752,1524610654,1524611555,152461245,1524613359,1524614261,1524615163,1524616065,1524616967,1524617869,1524618771,1524619673,1524620575,1524621477,1524622379,1524623281,1524624183,152462508,1524625987,1524626889,1524627790,1524628693,1524629594,1524630496,1524631398,1524632300,1524633202,1524634104,1524635006,1524635908,1524636810],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': [],
             'channel_0': True,
             'channel_1': False,
             'temperature': True
            },

            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15246',
             'patches_subdirectory': '/marion2018/patches_70MHz/15246/1524658039',
             'ctimes':[1524658039,1524658940,1524659842,1524660743,1524661645,1524662547,1524663449,1524664351,1524665253,1524666155,1524667057,1524667959,1524668861,152466976,1524670665,1524671567,1524672469,1524673375,1524674277,1524675179,1524676081,1524676983,1524677885,1524678787,1524679689,1524680590,1524681493,152468239,1524683296,1524684198,1524685100,1524686002,1524686904,1524687806,1524688708,1524689610,1524690512,1524691414,1524692316,1524693218,1524694120,152469502,1524695924,1524696825,1524697727,1524698629,1524699531],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': [],
             'channel_0': False,
             'channel_1': True,
             'temperature': True
            },
        ],

        15362:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15362',
             'patches_subdirectory': '/marion2018/patches_70MHz/15362/1536201211',
             'ctimes':[1536201211,1536201321,1536201834,1536202087,1536202915,1536203393,1536204347,1536205568,1536205721,1536205914,1536206510,1536207195,1536208524,1536209024,1536209686,1536209927,1536209950,1536210579,1536211295,1536211785,1536211831,1536212158,1536212251,1536212457,1536212865,1536213680,1536214339,1536214627,1536215270,1536215480,1536215802,1536216288,1536216753,1536217219,1536220825,1536224434,1536225686],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['offsetting.p'],
             'channel_0': True,
             'channel_1': False,
             'temperature': True
            },

            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15362',
             'patches_subdirectory': '/marion2018/patches_70MHz/15362/1536233063',
             'ctimes':[1536233063,1536234172,1536234844,1536235374,1536235929,1536244041,1536247648,1536249622,1536249988,1536250539,1536251099,1536251459,1536251621,153625201,1536252503,1536252968,1536253264,1536253426,1536254366,1536256810,1536257072,1536257606,1536259437,1536259923,1536260906,1536261237,1536261741,153626421,1536266625,1536267193,1536267658,1536269335,1536270120,1536272616,1536272726,1536273420,1536273604,1536274199,1536275987,1536276534,1536277245,153627761,1536278418,1536278442,1536278465,1536281514,1536289794,1536289895,1536291259,1536292005,1536292388,1536293205,1536293682,1536293957,1536294323,153629438,1536295801,1536295916,1536295966,1536296106,1536296402,1536297056],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['offsetting.p'],
             'channel_0': True,
             'channel_1': False,
             'temperature': True
            },
        ],
    },

    'switch':
    {
        15261:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'ctimes':[1526100607,1526101517,1526102419,1526103330,1526104231,1526105133,1526106045,1526106954,1526107856,1526108763,1526109670,1526110040,1526110103,152611015,1526110580,1526110817,1526111487,1526111722,1526112388,1526112632,1526113294,1526113539,1526114204,1526114446,1526115106,1526115313,1526115737,152611618,1526116243,1526116539,1526116598,1526116647,1526116955,1526117278,1526117518,1526118187,1526118423,1526118453,1526119094,1526119742,1526120186,152612109,1526121993,1526122900,1526123697,1526124602,1526125513,1526125724,1526126625,1526126834,1526127230,1526127293,1526127343,1526127393,1526127888,152612867,1526128909,1526129642,1526130543,1526131091,1526131423,1526132332,1526132388,1526132438,1526132765,1526132793,1526133717,1526134130,1526135031,152613593,1526136843,1526136873,1526137774,1526138684,1526139596,1526140501,1526141412,1526142313,1526143216,1526144127,1526145032,1526145299,1526146153,152614713,1526147515,1526148282,1526149002,1526149204,1526149493,1526149820,1526150527,1526151277,1526151449,1526152348,1526152644,1526153545,1526154452,152615535,1526156259,1526156881,1526157170,1526157263,1526157668,1526158573,1526159066,1526159717,1526159884,1526160305,1526161215,1526161512,1526162418,152616329,1526164052,1526164957,1526165859,1526166767,1526167677,1526168587,1526169493,1526169618,1526177060,1526177960,1526178872,1526179779,1526180681,152618158,1526182488,1526183391,1526184292,1526185199,1526185716,1526186617,1526187523,1526188434,1526189339,1526190247,1526191148,1526191955,1526192055,152619263,1526192964,1526193015,1526193078,1526193224,1526193540,1526193876,1526194449,1526194704,1526194782,1526195589,1526195687,1526196473,1526196594,152619737,1526197483,1526198281,1526198394,1526198906,1526199182],
             'scio_files': [('antenna.scio','antenna.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio'),('noise.scio','noise.scio')],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')]
            },
        ],
    },
}
