# NumPy
import numpy as np

# Copy
import copy

# Scio (https://github.com/sievers/scio)
import scio

# Date and Time
import time


def load_multiple_data(ctime_intervals, components, verbose=True):
    """ Loads PRIZM data associated with a given instrument component and ctime interval. """

    # Initializes the dictionary which will hold the data of interest.
    data = {
            component: {}
            for component in components
            }

    # Retrieves the metadata entries associated with the input ctime intervals and instrument components.
    metadata_entries = retrieve_multiple_metadata(ctime_intervals, components)

    # Loads and patches the PRIZM data associated with the input ctime intervals and instrument components.
    for metadata_entry in metadata_entries:
        # Loads and patches the PRIZM data associated with the current metadata entry.
        partial_data = load_data(metadata_entry, verbose)

        # Merges loaded and patched data with the data dictionary to be returned.
        data = merge_data(data, partial_data)

    return data


def merge_data(primary_data, secondary_data):
    """ Merges two PRIZM data dictionaries by concatenating all of its matching entries. """

    # Visits the the contents of each intrument component in the `secondary_data` dictionary.
    for component in secondary_data.keys():

        # Visits the contents of each data file associated with the current instrument component of the `secondary_data` dictionary.
        for file in secondary_data[component].keys():

            # Merges the `secondary_data` dictionary contents with those of the `primary_data` dictionary.
            if file in primary_data[component]:

                # Merges '.scio' files:
                if '.scio' in file:
                    primary_data[component][file] = np.vstack((primary_data[component][file], secondary_data[component][file]))

                # Merges '.raw' files:
                if '.raw' in file:
                    primary_data[component][file] = np.hstack((primary_data[component][file], secondary_data[component][file]))
            
            else:
                # Assimilates new files.
                primary_data[component][file] = secondary_data[component][file]

    return primary_data


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
    for file_name in metadata['patch_files']:
        data[component] = apply_patch(
                                      data[component],
                                      patches_subdirectory,
                                      file_name,
                                      verbose
                                      )

    return data


def apply_patch(data, patches_directory, file_name, verbose):
    """ Patches PRIZM data. """

    # Patches the data.
    if file_name == 'reordering.npy':
        # Loads the patch instructions.
        reordering = np.load(patches_directory + '/' + file_name, allow_pickle=True)

        # Rearranges the information contained in the input `data`.
        data['pol0.scio'] = data['pol0.scio'][reordering]
        data['pol1.scio'] = data['pol1.scio'][reordering]
        data['time_sys_start.raw'] = data['time_sys_start.raw'][reordering]
        data['time_sys_stop.raw'] = data['time_sys_stop.raw'][reordering]

    # Applies offsets to the `data` timestamps.
    if file_name == 'offsetting.npy':
        # Loads the patch instructions.
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
                # Verbose message.
                if verbose:
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

            # Stores the entries obtained above.
            for metadata_entry in metadata_entries:

                # Ensures no duplicates are stored:
                if metadata_entry not in retrieved_metadata:
                    retrieved_metadata.append(copy.deepcopy(metadata_entry))

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
             'ctimes': [1526100607,1526101517,1526102419,1526103330,1526104231,1526105133,1526106045,1526106954,1526107856,1526108763,1526109670,1526110040,1526110103,1526110153,1526110580,1526110817,1526111487,1526111722,1526112388,1526112632,1526113294,1526113539,1526114204,1526114446,1526115106,1526115313,1526115737,1526116180,1526116243,1526116539,1526116598,1526116647,1526116955,1526117278,1526117518,1526118187,1526118423,1526118453,1526119094,1526119742,1526120186,1526121092,1526121993,1526122900,1526123697,1526124602,1526125513,1526125724,1526126625,1526126834,1526127230,1526127293,1526127343,1526127393,1526127888,1526128678,1526128909,1526129642,1526130543,1526131091,1526131423,1526132332,1526132388,1526132438,1526132765,1526132793,1526133717,1526134130,1526135031,1526135933,1526136843,1526136873,1526137774,1526138684,1526139596,1526140501,1526141412,1526142313,1526143216,1526144127,1526145032,1526145299,1526146153,1526147136,1526147515,1526148282,1526149002,1526149204,1526149493,1526149820,1526150527,1526151277,1526151449,1526152348,1526152644,1526153545,1526154452,1526155353,1526156259,1526156881,1526157170,1526157263,1526157668,1526158573,1526159066,1526159717,1526159884,1526160305,1526161215,1526161512,1526162418,1526163298,1526164052,1526164957,1526165859,1526166767,1526167677,1526168587,1526169493,1526169618,1526177060,1526177960,1526178872,1526179779,1526180681,1526181587,1526182488,1526183391,1526184292,1526185199,1526185716,1526186617,1526187523,1526188434,1526189339,1526190247,1526191148,1526191955,1526192055,1526192632,1526192964,1526193015,1526193078,1526193224,1526193540,1526193876,1526194449,1526194704,1526194782,1526195589,1526195687,1526196473,1526196594,1526197374,1526197483,1526198281,1526198394,1526198906,1526199182],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': True,
             'polarization_1': True,
             'temperature': True
            },
        ],
    },

    '70MHz':
    {
        15245:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15245',
             'patches_subdirectory': '/marion2018/patches_70MHz/15236/1524500604',
             'ctimes': [1524500604,1524501506,1524502408,1524503310,1524504212,1524505114,1524506015,1524506918,1524507819,1524508721,1524509623,1524510525,1524511427,1524512329,1524513231,1524514133,1524515035,1524515937,1524516839,1524517741,1524518643,1524519545,1524520446,1524521348,1524522250,1524523152,1524524054,1524524956,1524525858,1524526760,1524527662,1524528564,1524529466,1524530368,1524531270,1524532172,1524533074,1524533976,1524534878,1524535780,1524536681,1524537583,1524538485,1524539387,1524540289,1524541191,1524542093,1524542995,1524543897,1524544799,1524545701,1524546603,1524547505,1524548407,1524549309,1524550211,1524551112,1524552015,1524552917,1524553818,1524554720,1524555622,1524556524,1524557426,1524558328,1524559230,1524560132,1524561034,1524561936,1524562838,1524563740,1524564642,1524565544,1524566446,1524567348,1524568249,1524569151,1524570053,1524570955,1524571857,1524572759,1524573661,1524574563,1524575465,1524576367,1524577269,1524578175,1524579077,1524579979,1524580881,1524581783,1524582685,1524583587,1524584489,1524585391,1524586293,1524587195,1524588096,1524588998,1524589900,1524590802,1524591709,1524592611,1524593512,1524594414,1524595321,1524596223,1524597124,1524598026,1524598928,1524599830],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15246:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15246',
             'patches_subdirectory': '/marion2018/patches_70MHz/15246/1524600732',
             'ctimes': [1524600732,1524601634,1524602536,1524603438,1524604341,1524605242,1524606144,1524607046,1524607948,1524608850,1524609752,1524610654,1524611555,1524612457,1524613359,1524614261,1524615163,1524616065,1524616967,1524617869,1524618771,1524619673,1524620575,1524621477,1524622379,1524623281,1524624183,1524625085,1524625987,1524626889,1524627790,1524628693,1524629594,1524630496,1524631398,1524632300,1524633202,1524634104,1524635006,1524635908,1524636810],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },

            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15246',
             'patches_subdirectory': '/marion2018/patches_70MHz/15246/1524658039',
             'ctimes': [1524658039,1524658940,1524659842,1524660743,1524661645,1524662547,1524663449,1524664351,1524665253,1524666155,1524667057,1524667959,1524668861,1524669763,1524670665,1524671567,1524672469,1524673375,1524674277,1524675179,1524676081,1524676983,1524677885,1524678787,1524679689,1524680590,1524681493,1524682394,1524683296,1524684198,1524685100,1524686002,1524686904,1524687806,1524688708,1524689610,1524690512,1524691414,1524692316,1524693218,1524694120,1524695022,1524695924,1524696825,1524697727,1524698629,1524699531],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': [],
             'polarization_0': False,
             'polarization_1': True,
             'temperature': True
            },
        ],

        15247:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15247',
             'patches_subdirectory': '/marion2018/patches_70MHz/15247/1524700433',
             'ctimes': [1524700433,1524701335,1524702237,1524703139,1524704041,1524704943,1524705845,1524706747,1524707649,1524708551,1524709453,1524710355,1524711256,1524712158,1524713060,1524713962,1524714864,1524715766,1524716668,1524717570,1524718472,1524719374,1524720276,1524721178,1524722080,1524722982,1524723884],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': ['trimming.npy'],
             'polarization_0': False,
             'polarization_1': True,
             'temperature': True
            },

            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15247',
             'patches_subdirectory': '/marion2018/patches_70MHz/15247/1524726386',
             'ctimes': [1524726386,1524727287,1524728194,1524729099,1524730006,1524730912,1524731818,1524732725,1524733631,1524734537,1524734592,1524735492,1524736399,1524737306,1524738211,1524739117,1524739721,1524740621,1524741529,1524742434,1524743336,1524744242,1524745148,1524746054,1524746961,1524747867,1524748773,1524749679,1524750586,1524751492,1524752398,1524753304,1524754211,1524755117,1524756023,1524756929,1524757836,1524758738,1524759645,1524760551,1524761456,1524762362,1524763269,1524764176,1524765081,1524765987,1524766893,1524767801,1524768706,1524769612,1524770519,1524771426,1524772331,1524773238,1524774143,1524775051,1524775956,1524776863,1524777768,1524778676,1524779581,1524780488,1524781393,1524782301,1524783206,1524784112,1524785018,1524785926,1524786831,1524787224,1524787247,1524794661,1524795562,1524796468,1524797374,1524798042,1524798197,1524798942,1524799098,1524799611,1524799848],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': False,
             'polarization_1': True,
             'temperature': True
            },
        ],

        15249:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15249',
             'patches_subdirectory': '/marion2018/patches_70MHz/15249/1524900809',
             'ctimes': [1524900809,1524901715,1524902621,1524903529,1524904435,1524905340,1524906246,1524907154,1524908060,1524908965,1524909871,1524910779,1524911684,1524912590,1524913496,1524914404,1524915309,1524916211,1524917117,1524918024,1524918929,1524919625,1524920525,1524921431,1524922339,1524923244,1524924150,1524925056,1524925964,1524926869,1524927775,1524928681,1524929589,1524930494,1524931400,1524932306,1524933214],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['offsetting.npy'],
             'polarization_0': False,
             'polarization_1': True,
             'temperature': True
            },
        ],

        15357:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15357',
             'patches_subdirectory': '/marion2018/patches_70MHz/15357/1535726612',
             'ctimes': [1535726612,1535730223,1535733828,1535737434,1535741034,1535742004,1535745607,1535749210,1535752550,1535752912,1535754016,1535755470,1535756625,1535756675,1535757338,1535757535,1535757758,1535758689,1535758800,1535758976,1535759276,1535759503,1535759635,1535759810,1535760015,1535760038,1535770819,1535770960,1535771352,1535772335,1535773768,1535774786,1535775152,1535776991,1535777261,1535778542,1535778808,1535780782,1535781489,1535781811,1535782670,1535782923,1535783578,1535784415,1535784495,1535785037,1535785207,1535785642,1535789244,1535792852,1535794827,1535794890,1535795156,1535795871,1535796086,1535796110,1535796371,1535796492,1535796767,1535796926,1535797245,1535797360,1535797410,1535797610,1535797966,1535798062,1535798483,1535798541,1535799093,1535799130,1535799531,1535799638],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15362:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15362',
             'patches_subdirectory': '/marion2018/patches_70MHz/15362/1536201211',
             'ctimes': [1536201211,1536201321,1536201834,1536202087,1536202915,1536203393,1536204347,1536205568,1536205721,1536205914,1536206510,1536207195,1536208524,1536209024,1536209686,1536209927,1536209950,1536210579,1536211295,1536211785,1536211831,1536212158,1536212251,1536212457,1536212865,1536213680,1536214339,1536214627,1536215270,1536215480,1536215802,1536216288,1536216753,1536217219,1536220825,1536224434,1536225686],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },

            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15362',
             'patches_subdirectory': '/marion2018/patches_70MHz/15362/1536233063',
             'ctimes': [1536233063,1536234172,1536234844,1536235374,1536235929,1536244041,1536247648,1536249622,1536249988,1536250539,1536251099,1536251459,1536251621,1536252013,1536252503,1536252968,1536253264,1536253426,1536254366,1536256810,1536257072,1536257606,1536259437,1536259923,1536260906,1536261237,1536261741,1536264214,1536266625,1536267193,1536267658,1536269335,1536270120,1536272616,1536272726,1536273420,1536273604,1536274199,1536275987,1536276534,1536277245,1536277610,1536278418,1536278442,1536278465,1536281514,1536289794,1536289895,1536291259,1536292005,1536292388,1536293205,1536293682,1536293957,1536294323,1536294386,1536295801,1536295916,1536295966,1536296106,1536296402,1536297056],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15371:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15371',
             'patches_subdirectory': '/marion2018/patches_70MHz/15371/1537108254',
             'ctimes': [1537108254,1537111858,1537115464,1537119073,1537120066,1537123669,1537127274,1537130881,1537133239,1537133592,1537133625,1537134119,1537137729,1537137885,1537138704,1537139213,1537139431,1537141832,1537142236,1537142325,1537142417,1537142441,1537153222,1537153684,1537153825,1537153910,1537154116,1537154140,1537155153,1537155441,1537155530,1537156007,1537156684,1537156829,1537157096,1537157200,1537157884,1537158788,1537158830,1537159000,1537160247,1537160755,1537161393,1537161489,1537162416,1537162933,1537162963,1537163674,1537163926,1537164443,1537165068,1537168670,1537172277,1537173553,1537175245,1537175390,1537175643,1537175900,1537176473,1537176545,1537176811,1537178296,1537178999,1537179045,1537180564,1537180881,1537181077,1537183444,1537183555,1537183605,1537184269,1537184513,1537185445,1537185892,1537186418,1537186671,1537186868,1537187030,1537187085,1537187381,1537188361,1537191497,1537191931,1537192297,1537195907,1537199511],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15384:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15384',
             'patches_subdirectory': '/marion2018/patches_70MHz/15384/1538404716',
             'ctimes': [1538404716,1538405354,1538405391,1538405441,1538405581,1538406046,1538408279,1538409794,1538410605,1538411295,1538412089,1538412347,1538412773,1538412961,1538413630,1538414794,1538414956,1538415348,1538415658,1538416486,1538419124,1538419930,1538423537,1538427143,1538430748,1538434350,1538437955,1538438420,1538438444,1538438467,1538442068,1538445671,1538449271,1538452875,1538455091,1538455391,1538456024,1538456800,1538457887,1538458361,1538458476,1538458526,1538459189,1538459261,1538460339,1538460441,1538460491,1538461462,1538462039,1538463195,1538466798,1538470405,1538474010,1538477611,1538481216,1538484823,1538485448,1538485516,1538487775,1538487972,1538489301,1538489416,1538489466,1538489666,1538490455,1538491042,1538491131,1538491181,1538491538,1538491571,1538495072,1538495890,1538496113,1538496136,1538496524,1538497064,1538497343,1538497799,1538497867,1538497977,1538498353,1538498875,1538499369],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15395:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15395',
             'patches_subdirectory': '/marion2018/patches_70MHz/15395/1539527455',
             'ctimes': [1539527455,1539531066,1539532237,1539532360,1539532747,1539536352,1539539953,1539543214,1539543554,1539544520,1539545002,1539545744,1539546404,1539546771,1539547323,1539547360,1539547687,1539547931,1539548774,1539549252,1539549380,1539550199,1539550919,1539552421,1539552510,1539554612,1539555415,1539555495,1539555868,1539556704,1539557213,1539557280,1539557576,1539557991,1539558050,1539559388,1539559520,1539561628,1539569119,1539569472,1539570269,1539570410,1539570832,1539572745,1539573015,1539573799,1539574549,1539575576,1539577397,1539578320,1539578405,1539579182,1539580201,1539580350,1539582785,1539585504,1539585982,1539589589,1539589675,1539590248,1539591175,1539594780,1539597287,1539597355,1539597621,1539597987,1539598604,1539598999,1539599689],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15404:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15404',
             'patches_subdirectory': '/marion2018/patches_70MHz/15404/1540480552',
             'ctimes': [1540480552,1540484163,1540486501,1540486547,1540489414,1540489940,1540490330,1540490829,1540491350,1540491633,1540491908,1540492412,1540492527,1540492577,1540493145,1540493242,1540493430,1540493561,1540494190,1540494232,1540494432,1540497540,1540498341,1540499951],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15419:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15419',
             'patches_subdirectory': '/marion2018/patches_70MHz/15419/1541949307',
             'ctimes': [1541949307,1541952917,1541955498,1541955521,1541955778,1541957215,1541958562,1541958979,1541959626,1541960406,1541961252,1541963432,1541964220,1541964599,1541964843,1541964927,1541965175,1541965337,1541966373,1541967106,1541970708,1541974312,1541975997,1541976372,1541976673,1541977129,1541977317,1541978136,1541978782,1541979420,1541979799,1541980825,1541980847,1541991681,1541992984,1541993324,1541995223,1541995407,1541998081,1541998222,1541998337,1541998482,1541999181],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15428:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15428',
             'patches_subdirectory': '/marion2018/patches_70MHz/15428/1542896208',
             'ctimes': [1542896208,1542896836,1542897742,1542899019,1542899082],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15443:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15443',
             'patches_subdirectory': '/marion2018/patches_70MHz/15443/1544366096',
             'ctimes': [1544366096,1544367793,1544368024,1544369456,1544370509,1544370705,1544371447,1544371890,1544371953,1544372923,1544374404,1544375167,1544375700,1544375949,1544375973,1544376835,1544377240,1544377856,1544378879,1544379119,1544379282,1544382889,1544386492,1544387745,1544388392,1544388528,1544388885,1544390465,1544392218,1544392519,1544393071,1544394979,1544395729,1544397936,1544398129,1544398369,1544398700],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15453:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15453',
             'patches_subdirectory': '/marion2018/patches_70MHz/15453/1545320146',
             'ctimes': [1545320146,1545322340,1545323021,1545323892,1545324309,1545326599,1545326757,1545327179,1545327458,1545327668,1545328721,1545330697,1545331149,1545331338,1545331945,1545332275,1545334556,1545334813,1545336220,1545337243,1545337868,1545338264,1545338815,1545338942,1545339389,1545339688,1545339903,1545341275,1545341563,1545342206,1545342910,1545343072,1545343494,1545344664,1545345492,1545345710,1545345812,1545347396,1545348911,1545350425,1545357918,1545360193,1545361082,1545361227,1545361282,1545361521,1545361915,1545363534,1545363939,1545364062,1545364207,1545364599,1545364972,1545365047,1545365162,1545365419,1545365728,1545366214,1545368678,1545369718,1545370572,1545371346,1545371487,1545371542,1545372326,1545372523,1545373913,1545374128,1545374886,1545376925,1545380535,1545384139,1545387741,1545391354,1545391427,1545391768,1545395376,1545398977],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15469:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15469',
             'patches_subdirectory': '/marion2018/patches_70MHz/15469/1546960553',
             'ctimes': [1546960553,1546964160,1546964549,1546965074,1546966273,1546967287,1546967816,1546967930,1546968252,1546968311,1546969186,1546969621,1546971945,1546972237,1546973000,1546973136,1546973938,1546974343,1546974622,1546974801,1546975067,1546976336,1546977125,1546978061,1546979031,1546979664,1546980332,1546981035,1546981262,1546981411,1546982045,1546982382,1546982471,1546982611,1546982661,1546983168,1546983955,1546984433,1546984501,1546984767,1546984826,1546985262,1546986151,1546989756,1546992030,1546992054,1546995648],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15485:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15485',
             'patches_subdirectory': '/marion2018/patches_70MHz/15485/1548512465',
             'ctimes': [1548512465,1548514482,1548514550,1548515184,1548515350,1548515881,1548516277,1548516460,1548517400,1548517450,1548517729,1548518449,1548520240,1548522564,1548522977,1548523978,1548524170,1548524471,1548525053,1548525155,1548526017,1548526759,1548526956,1548527070,1548527487,1548527904,1548528015,1548529172,1548529235,1548529435,1548529877,1548530825,1548531212,1548531672,1548531800,1548532728,1548533016,1548534332,1548534797,1548534865,1548535576,1548536465,1548537345,1548537460,1548541067,1548544674,1548545434,1548545772,1548546851,1548547217,1548547240,1548547263,1548557656,1548558545,1548559027,1548559306,1548559485,1548559861,1548560413,1548560968,1548561986,1548562130,1548562383,1548563989,1548564649,1548568259,1548571864,1548575466,1548576084,1548576562,1548576690,1548576986,1548577568,1548578388,1548581994,1548583646,1548585421,1548585607,1548585752,1548588109,1548588829,1548589214,1548589305,1548589890,1548591111,1548591965,1548592472,1548593088,1548594587,1548595510,1548595957,1548596080,1548596743,1548597844,1548597890],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15498:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15498',
             'patches_subdirectory': '/marion2018/patches_70MHz/15498/1549812840',
             'ctimes': [1549812840,1549815651,1549815955,1549817272,1549817555,1549818891,1549819326,1549820189,1549820666,1549821071,1549822162,1549822293,1549823163,1549823365,1549824520,1549824902,1549825095,1549825576,1549826072,1549826416,1549826896,1549828493,1549828569,1549829203,1549829767,1549830007,1549833610,1549837216,1549837913,1549838326,1549838545,1549840103,1549840286,1549841365,1549841550,1549843169,1549843218,1549850662,1549850967,1549851408,1549851665,1549851689,1549851899,1549852046,1549854097,1549854306,1549854385,1549854600,1549855229,1549855676,1549856202,1549856696,1549860299,1549863901,1549867510,1549870903,1549871256,1549871621,1549875231,1549878835,1549882447,1549883691,1549884277,1549884550,1549884765,1549884879,1549885085,1549885259,1549885665,1549886031,1549886184,1549886787,1549886824,1549887381,1549887630,1549888634,1549888689,1549889840,1549890867,1549891342,1549891504,1549891800,1549892593,1549892946,1549893118,1549893194,1549894074,1549894340,1549895519,1549895634,1549896328,1549896585,1549896825,1549896987,1549897149,1549897415,1549897715,1549898021],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15505:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15505',
             'patches_subdirectory': '/marion2018/patches_70MHz/15505/1550503727',
             'ctimes': [1550503727,1550505048,1550505145,1550505532,1550506359,1550508078,1550509220,1550509275,1550509571,1550509690,1550510184,1550511337,1550511490,1550511545,1550512515,1550516079,1550517106,1550518237,1550518270,1550519789,1550520432,1550520650,1550521711,1550521830,1550522524,1550523287,1550523722,1550524278,1550525236,1550525415,1550525885,1550526800,1550530405,1550531703,1550531986,1550532503,1550532738,1550534345,1550534416,1550534440,1550534463,1550546590,1550547198,1550547330,1550547505,1550547710,1550548342,1550548427,1550548497,1550548962,1550549622,1550552574,1550553207,1550556176,1550556809,1550557780,1550557950,1550557997,1550558150,1550558682,1550559937,1550560000,1550560140,1550560466,1550561091,1550561457,1550564705,1550565001,1550565427,1550565718,1550569161,1550569743,1550569905,1550569960,1550570420,1550570565,1550570650,1550571560,1550571796,1550571958,1550572215,1550573276,1550573858,1550574604,1550575771,1550575860,1550576981,1550577256,1550577959,1550578451,1550578635,1550579304,1550579621,1550580047,1550580436,1550580585,1550581284,1550581697,1550581976,1550582390,1550582872,1550585944,1550586206,1550586909,1550587611,1550587699,1550587956,1550587980,1550591245,1550591772,1550592544,1550593143,1550594464,1550594651,1550595122,1550596162,1550597046,1550597403,1550598321],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15522:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15522',
             'patches_subdirectory': '/marion2018/patches_70MHz/15522/1552223453',
             'ctimes': [1552223453,1552224479,1552225299,1552227185,1552228375,1552229276,1552229374,1552229550,1552230031,1552230094,1552233840,1552233928,1552234678,1552235464,1552235665,1552236690,1552237432,1552237815,1552238059,1552238083,1552239128,1552240587,1552244193,1552247804,1552251413,1552252203,1552253109,1552253526,1552253847,1552253949,1552253999,1552254169,1552255512,1552255686,1552256120,1552256455,1552257284,1552257671,1552258090,1552261696,1552262297,1552262415,1552262439,1552262462,1552277369,1552278218,1552279793,1552280176,1552283782,1552284479,1552287386,1552287916,1552288090,1552288321,1552289245,1552289455,1552290213,1552290993,1552291703,1552293991,1552294601,1552295317,1552297238,1552299643],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15540:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15540',
             'patches_subdirectory': '/marion2018/patches_70MHz/15540/1554041600',
             'ctimes': [1554041600,1554041962,1554042086,1554042116,1554042632,1554043543,1554043982,1554044749,1554044951,1554045035,1554046083,1554047689,1554049545,1554049877,1554053484,1554057088,1554060694,1554064305,1554067909,1554071513,1554075125,1554076828,1554084321,1554086836,1554087357,1554087928,1554088251,1554088466,1554089592,1554090959,1554090968,1554094562,1554098169],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],

        15553:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15553',
             'patches_subdirectory': '/marion2018/patches_70MHz/15553/1555335072',
             'ctimes': [1555335072,1555336372,1555338919,1555339613,1555340238,1555340479,1555341363,1555341721,1555343132,1555343601,1555344050,1555347657,1555351259,1555351755,1555351934,1555352170,1555352319,1555352544,1555352996,1555353365,1555356976,1555357451,1555357639,1555358259,1555358710,1555358919,1555359064,1555359360,1555359617,1555363223,1555364204,1555364843,1555365070,1555365094,1555365282,1555365444,1555365740,1555366045,1555367301,1555367598,1555368076,1555368144,1555368410,1555370335,1555370653,1555372829,1555372853,1555376463,1555380066,1555383675,1555384750,1555385392,1555385969,1555389572,1555393174,1555396783,1555397176,1555397420,1555397997,1555398033,1555399047,1555399335],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': True,
             'polarization_1': False,
             'temperature': True
            },
        ],
    },

    'switch':
    {
        15245:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524501952],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524502685],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524509170],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524510275],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524516384],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524517865],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524523599],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524525455],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524530818],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524533045],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524538029],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524540635],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524545247],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524548225],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524552465],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524555815],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524559682],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524563405],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524566901],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524570995],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524574119],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524578585],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524581341],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524586175],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524588556],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524593765],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15245',
             'patches_subdirectory': '',
             'ctimes': [1524595778],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },
        ],

        15246:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524601355],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524602997],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524608944],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524610214],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524616534],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524617436],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524624123],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524624656],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524631713],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524631870],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524644691],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524644693],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524658023],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524658025, 1524665245],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524665613],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524672470],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524673203],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524679686],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524680793],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524686902],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524688383],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524694120],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15246',
             'patches_subdirectory': '',
             'ctimes': [1524695973],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15247:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524701338],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524703563],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524708552],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524711153],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524715766],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524718743],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524722987],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524726333],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524730209],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524733924],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524737424],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524741513],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524744641],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524749103],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524751856],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524756693],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524759075],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524764283],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524766288],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524771873],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524773508],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524779463],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524780733],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524787053],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524787955],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524794643],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15247',
             'patches_subdirectory': '',
             'ctimes': [1524795173],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },
        ],

        15249:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15249',
             'patches_subdirectory': '',
             'ctimes': [1524900895],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15249',
             'patches_subdirectory': '',
             'ctimes': [1524903494],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15249',
             'patches_subdirectory': '',
             'ctimes': [1524908485],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15249',
             'patches_subdirectory': '',
             'ctimes': [1524910716],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15249',
             'patches_subdirectory': '',
             'ctimes': [1524916075],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15249',
             'patches_subdirectory': '',
             'ctimes': [1524917934],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },
        ],

        15357:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535710618],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535710620],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535712748],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535712750],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535713043],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535713045],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535715835],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535715837],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535726602],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535726604,1535733845],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535734605],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535741088],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535742608],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535748328],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535750612],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535755572],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15357',
             'patches_subdirectory': '',
             'ctimes': [1535758615,1535766618,1535774622,1535782625,1535790628,1535798632],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15362:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536206803,1536214806,1536222809,1536244042],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536244044,1536251283],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536252045],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536258527],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536260049],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536265769],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536268052],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536273009],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15362',
             'patches_subdirectory': '',
             'ctimes': [1536276055,1536284059,1536292062],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15371:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537108244],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537108246,1537115487],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537116247],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537122727],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537124249],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537129968],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537132252],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537137208],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15371',
             'patches_subdirectory': '',
             'ctimes': [1537140255,1537148258,1537156261,1537164264,1537172267,1537180270,1537188272,1537196275],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15384:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538404709],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538404711,1538411956],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538412712],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538419194],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538420715],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538426435],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538428718],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538433676],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15384',
             'patches_subdirectory': '',
             'ctimes': [1538436722,1538444725,1538452728,1538460732,1538468735,1538476738,1538484742,1538492745],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15395:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539527444],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539527446,1539534687],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539535448],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539541930],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539543451],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539549176],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539551454],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539556413],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15395',
             'patches_subdirectory': '',
             'ctimes': [1539559458,1539567461,1539575464,1539583468,1539591471,1539599474],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15404:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15404',
             'patches_subdirectory': '',
             'ctimes': [1540480542],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15404',
             'patches_subdirectory': '',
             'ctimes': [1540480544,1540487786],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15404',
             'patches_subdirectory': '',
             'ctimes': [1540488545],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15404',
             'patches_subdirectory': '',
             'ctimes': [1540495028],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15404',
             'patches_subdirectory': '',
             'ctimes': [1540496548],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15419:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15419',
             'patches_subdirectory': '',
             'ctimes': [1541949297],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15419',
             'patches_subdirectory': '',
             'ctimes': [1541949299],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15419',
             'patches_subdirectory': '',
             'ctimes': [1541957300,1541965303,1541973307,1541981310,1541989313,1541997317],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15428:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15428',
             'patches_subdirectory': '',
             'ctimes': [1542896198],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15428',
             'patches_subdirectory': '',
             'ctimes': [1542896200],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },
        ],

        15443:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15443',
             'patches_subdirectory': '',
             'ctimes': [1544366084],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15443',
             'patches_subdirectory': '',
             'ctimes': [1544366086, 1544373340],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15443',
             'patches_subdirectory': '',
             'ctimes': [1544374087],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15443',
             'patches_subdirectory': '',
             'ctimes': [1544380598],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15443',
             'patches_subdirectory': '',
             'ctimes': [1544382090],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15443',
             'patches_subdirectory': '',
             'ctimes': [1544387851],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15443',
             'patches_subdirectory': '',
             'ctimes': [1544390094,1544398097],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15453:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15453',
             'patches_subdirectory': '',
             'ctimes': [1545320137],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15453',
             'patches_subdirectory': '',
             'ctimes': [1545320139,1545327392],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15453',
             'patches_subdirectory': '',
             'ctimes': [1545328140,1545336144,1545344147,1545352150,1545360154,1545368157,1545376160,1545384164,1545392167],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15469:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546960543],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546960545,1546967802],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546968546],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546975058],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546976550],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546982313],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546984553],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546989568],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15469',
             'patches_subdirectory': '',
             'ctimes': [1546992556],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15485:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15485',
             'patches_subdirectory': '',
             'ctimes': [1548512453],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15485',
             'patches_subdirectory': '',
             'ctimes': [1548512455,1548519717],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15485',
             'patches_subdirectory': '',
             'ctimes': [1548520456,1548528459,1548536463,1548544466,1548552470,1548560473,1548568476,1548576480,1548584483,1548592487],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15498:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549812828],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549812830,1549820082],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549820831],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549827341],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549828834],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549834594],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549836838],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549841851],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15498',
             'patches_subdirectory': '',
             'ctimes': [1549844841,1549852844,1549860848,1549868851,1549876854,1549884858,1549892861],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15505:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550503719],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550503721,1550510977],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550511722],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550518234],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550519725],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550525483],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550527729],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550532735],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550535732],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550539983],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550543735],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550547232],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550551739],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550554487],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550559742],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550561748],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550567745],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550569007],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550575749],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550576268,1550583523],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550583752],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550590774],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550591755],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550598032],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15505',
             'patches_subdirectory': '',
             'ctimes': [1550599759],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15522:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15522',
             'patches_subdirectory': '',
             'ctimes': [1552223439],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15522',
             'patches_subdirectory': '',
             'ctimes': [1552223441],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15522',
             'patches_subdirectory': '',
             'ctimes': [1552231442,1552239446,1552247449,1552255452,1552263456,1552271459,1552279462,1552287466,1552295469],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15540:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554041589],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554041591,1554048839],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554049593],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554056083],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554057596],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554063325],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554065599],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554070560],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15540',
             'patches_subdirectory': '',
             'ctimes': [1554073603,1554081606,1554089609,1554097613],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15553:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555335060],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555335062,1555342314],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555343064],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555349561],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555351067],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555356809],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555359071],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555364053],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555367074],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555371304],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15553',
             'patches_subdirectory': '',
             'ctimes': [1555375077,1555383081,1555391084,1555399087],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],
    },
}
