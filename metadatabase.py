# NumPy
import numpy as np

# Copy
import copy

# Scio (https://github.com/sievers/scio)
import scio

# Date and Time
import time


def load_multiple_data(ctime_intervals, components, filters=[], patch=True, verbose=True):
    """ Loads PRIZM data associated with a given instrument component and ctime interval. """

    # Initializes the dictionary which will hold the data of interest.
    data = {
            component: {}
            for component in components
            }

    # Retrieves the metadata entries associated with the input ctime intervals and instrument components.
    metadata_entries = retrieve_multiple_metadata(ctime_intervals, components, filters)

    # Loads and patches the PRIZM data associated with the input ctime intervals and instrument components.
    for metadata_entry in metadata_entries:
        # Loads and patches the PRIZM data associated with the current metadata entry.
        partial_data = load_data(metadata_entry, patch, verbose)

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
                    # Checks whether any of the arrays being stacked are empty in order to avoid dimensionality errors.
                    if len(primary_data[component][file]) == 0:
                        primary_data[component][file] = secondary_data[component][file]
                    elif len(secondary_data[component][file]) != 0:
                        primary_data[component][file] = np.vstack((primary_data[component][file], secondary_data[component][file]))

                # Merges '.raw' files:
                if '.raw' in file:
                    primary_data[component][file] = np.hstack((primary_data[component][file], secondary_data[component][file]))

            else:
                # Assimilates new files.
                primary_data[component][file] = secondary_data[component][file]

    return primary_data


def load_data(metadata, patch=True, verbose=True):
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

    if patch:
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


def retrieve_multiple_metadata(ctime_intervals, components, filters=[]):
    """ Retrieves the PRIZM metadata associated with multiple instrument components and ctime intervals. """

    # Initializes the list which will hold the `metadata` entries of interest.
    retrieved_metadata = []

    # For each input `component` ...
    for component in components:

        # And each `initial_ctime` and `final_ctime` in the input `ctime_intervals` ...
        for initial_ctime, final_ctime in ctime_intervals:

            # Retrieves the `metadata_entries` of interest.
            metadata_entries = retrieve_metadata(initial_ctime, final_ctime, component, filters)

            # Stores the entries obtained above.
            for metadata_entry in metadata_entries:

                # Ensures no duplicates are stored:
                if metadata_entry not in retrieved_metadata:
                    retrieved_metadata.append(copy.deepcopy(metadata_entry))

    # Returns a list containing the desired `metadata` entries.
    return retrieved_metadata


def retrieve_metadata(initial_ctime, final_ctime, component, filters=[]):
    """ Retrieves the PRIZM metadata associated with a given instrument component and ctime interval. """

    # Checks whether `initial_ctime` < `final_ctime`. If not, the values of these inputs are swapped.
    if initial_ctime > final_ctime:
        initial_ctime, final_ctime = final_ctime, initial_ctime

    # Extracts the `metadatabase` keys associated with the input ctime range.
    initial_key = int(str(initial_ctime)[:5])
    final_key = int(str(final_ctime)[:5])

    # Collects all `metadatabase` entries associated with the input ctime range.
    retrieved_metadata = [
                          metadata_entry
                          for key in metadatabase[component].keys()
                          for metadata_entry in metadatabase[component][key]
                          if key >= initial_key and key < final_key
                          ]

    # Apply the input filters.
    retrieved_metadata = filter_metadata(retrieved_metadata, filters)

    # Returns a list containing the desired `metadata` entries.
    return retrieved_metadata


def filter_metadata(metadata, filters=[]):
    """ Discards antenna metadata entries lacking the fields listed in the input filter list. """
    
    # The `metadata` entries to be discarded.
    filtered_metadata = [
                         metadata_entry
                         for metadata_entry in metadata
                         if metadata_entry['component'] == 'switch'
                         or all(len(metadata_entry[field]) != 0 for field in filters)
                         ]

    return filtered_metadata


# User-defined directories.
data_directory = '/Volumes/PRIZM_DISK1'
patches_directory = '/Users/Fernando/Documents/Code/Repositories/prizmatoid/patches_data'


# Metadatabase.
metadatabase = {
    '100MHz':
    {
        15244:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15244',
             'patches_subdirectory': '/marion2018/patches_100MHz/15244/1524485407',
             'ctimes':[1524485407,1524486308,1524487210,1524488113,1524489019,1524489920,1524490823,1524491724,1524492626,1524493528,1524494430,1524495332,1524496234,1524497136,1524498038,1524498940,1524499842],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': ['trimming.npy'],
             'polarization_0': [1524485412,1524500742],
             'polarization_1': [1524485412,1524500742],
             'temperature': True
             },
        ],

        15247:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15247',
             'patches_subdirectory': '/marion2018/patches_100MHz/15247/1524700435',
             'ctimes': [1524700435,1524701337,1524702239,1524703141,1524704043,1524704945,1524705847,1524706749,1524707651,1524708553,1524709455,1524710357,1524711259,1524712161,1524713062,1524713964,1524714867,1524715768,1524716670,1524717572,1524718474,1524719376,1524720278,1524721180,1524722082,1524722984,1524723886,1524724788],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': [],
             'polarization_0': [1524700452,1524725562],
             'polarization_1': [1524700452,1524725562],
             'temperature': True
            },

            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15247',
             'patches_subdirectory': '/marion2018/patches_100MHz/15247/1524726761',
             'ctimes': [1524726761,1524727322,1524728223,1524729129,1524730035,1524730942,1524731753,1524732283,1524733183,1524734090,1524734998,1524735903,1524736657,1524737558,1524737890,1524738790,1524739697,1524740604,1524741092,1524741992,1524742898,1524743805,1524744712,1524745618,1524746524,1524747430,1524748337,1524749243,1524750148,1524750269,1524751170,1524752076,1524752248,1524753150,1524754055,1524754962,1524755868,1524756775,1524757681,1524758587,1524759494,1524760228,1524761130,1524762036,1524762942,1524763848,1524764754,1524765660,1524766568,1524767473,1524768380,1524769286,1524770192,1524771098,1524772005,1524772911,1524773817,1524774723,1524775629,1524776532,1524776561,1524777462,1524778368,1524779274,1524780180,1524781087,1524781992,1524782899,1524783805,1524784712,1524784987,1524785888,1524786794,1524787225],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['trimming.npy'],
             'polarization_0': [1524726771,1524787198],
             'polarization_1': [1524726771,1524787198],
             'temperature': True
            },
        ],

        15252:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15252',
             'patches_subdirectory': '/marion2018/patches_100MHz/15252/1525200770',
             'ctimes': [1525200770,1525201114,1525202020,1525202273,1525203182,1525203951,1525204430,1525204744,1525205223,1525205835,1525206741,1525207642,1525208074,1525208975,1525209087,1525209449,1525210176,1525210438,1525211297,1525211689,1525211787,1525211983,1525212072,1525212981,1525213313,1525214081,1525214343,1525215248,1525215425,1525216326,1525217232,1525218134,1525219036,1525219221,1525219244,1525219268,1525220170,1525221071,1525221973,1525222880,1525223790,1525224696,1525225598,1525226487,1525227392,1525228299,1525229205,1525230115,1525231100,1525232001,1525232912,1525233823,1525234733,1525235635,1525236537,1525237447,1525238357,1525239260,1525240166,1525241072,1525241975,1525242414,1525242512,1525243417,1525243880,1525244038,1525244784,1525244943,1525244967,1525245752,1525246043,1525246944,1525247854,1525248074,1525248137,1525248433,1525248522,1525248637,1525248982,1525249887,1525250103,1525251005,1525251915,1525252787,1525252837,1525253687,1525254001,1525254233,1525255134,1525255544,1525256437,1525256643,1525257208,1525257297,1525257930,1525258097,1525258182,1525259090,1525259411,1525259487,1525260396,1525260915,1525261078,1525261102,1525261351,1525261427,1525262151,1525262499,1525262562,1525263366,1525263627,1525264532,1525265440,1525266307,1525266820,1525267730,1525268631,1525269422,1525269978,1525270067,1525270333,1525270512,1525271090,1525272066,1525272972,1525273877,1525274780,1525275526,1525276427,1525277338,1525278240,1525279141,1525280049,1525280600,1525281501,1525282411,1525283318,1525284225,1525284292,1525284408,1525285059,1525285736,1525286240,1525286704,1525286983,1525287241,1525287317,1525288226,1525289133,1525290039,1525290293,1525290382,1525290774,1525291409,1525291568,1525291592,1525292165,1525293066,1525293977,1525294880,1525295781,1525296688,1525297598,1525297934,1525298840,1525299141,1525299217],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': [1525200785,1525300118],
             'polarization_1': [1525200785,1525300118],
             'temperature': False
             },
        ],

        15260:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15260',
             'patches_subdirectory': '/marion2018/patches_100MHz/15260/1526046765',
             'ctimes': [1526046765,1526047382,1526047483,1526048389,1526049188,1526050093,1526050430,1526051331,1526052232,1526053144,1526054046,1526054947,1526055854,1526056758,1526057662,1526058566,1526059035,1526059345,1526059710,1526060140,1526060342,1526060755,1526060974,1526061883,1526062315,1526063217,1526063756,1526064665,1526064849,1526064938,1526065204,1526065383,1526066172,1526067078,1526067549,1526068453,1526069365,1526069956,1526070184,1526071008,1526071395,1526071600,1526071970,1526072124,1526072606,1526072703,1526073608,1526073933,1526074355,1526074881,1526075782,1526076236,1526076489,1526077394,1526078031,1526078932,1526079834,1526080736,1526081646,1526082548,1526083221,1526083245,1526083269,1526084175,1526085086,1526085213,1526094558,1526095467,1526096378,1526096991,1526097893,1526098795,1526099705],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['trimming.npy'],
             'polarization_0': [1526046776,1526100606],
             'polarization_1': [1526046776,1526100606],
             'temperature': True
             },
        ],

        15261:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15261',
             'patches_subdirectory': '/marion2018/patches_100MHz/15261/1526100607',
             'ctimes': [1526100607,1526101517,1526102419,1526103330,1526104231,1526105133,1526106045,1526106954,1526107856,1526108763,1526109670,1526110040,1526110103,1526110153,1526110580,1526110817,1526111487,1526111722,1526112388,1526112632,1526113294,1526113539,1526114204,1526114446,1526115106,1526115313,1526115737,1526116180,1526116243,1526116539,1526116598,1526116647,1526116955,1526117278,1526117518,1526118187,1526118423,1526118453,1526119094,1526119742,1526120186,1526121092,1526121993,1526122900,1526123697,1526124602,1526125513,1526125724,1526126625,1526126834,1526127230,1526127293,1526127343,1526127393,1526127888,1526128678,1526128909,1526129642,1526130543,1526131091,1526131423,1526132332,1526132388,1526132438,1526132765,1526132793,1526133717,1526134130,1526135031,1526135933,1526136843,1526136873,1526137774,1526138684,1526139596,1526140501,1526141412,1526142313,1526143216,1526144127,1526145032,1526145299,1526146153,1526147136,1526147515,1526148282,1526149002,1526149204,1526149493,1526149820,1526150527,1526151277,1526151449,1526152348,1526152644,1526153545,1526154452,1526155353,1526156259,1526156881,1526157170,1526157263,1526157668,1526158573,1526159066,1526159717,1526159884,1526160305,1526161215,1526161512,1526162418,1526163298,1526164052,1526164957,1526165859,1526166767,1526167677,1526168587,1526169493,1526169618,1526177060,1526177960,1526178872,1526179779,1526180681,1526181587,1526182488,1526183391,1526184292,1526185199,1526185716,1526186617,1526187523,1526188434,1526189339,1526190247,1526191148,1526191955,1526192055,1526192632,1526192964,1526193015,1526193078,1526193224,1526193540,1526193876,1526194449,1526194704,1526194782,1526195589,1526195687,1526196473,1526196594,1526197374,1526197483,1526198281,1526198394,1526198906,1526199182],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': [1526100623,1526200089],
             'polarization_1': [1526100623,1526200089],
             'temperature': True
            },
        ],

        15262:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15262',
             'patches_subdirectory': '/marion2018/patches_100MHz/15262/1526200089',
             'ctimes': [1526200089,1526200990,1526201340,1526201543,1526201996,1526202310,1526202343,1526212237,1526212793,1526213474,1526213676,1526213700,1526214595,1526215259,1526215366,1526215510,1526216149,1526216671,1526217434,1526218244,1526219149,1526219205,1526219868,1526220769,1526221241,1526221330,1526221657,1526222562,1526223244,1526223290,1526224190,1526225101,1526226007,1526226914,1526227815,1526228726,1526229632,1526230534,1526231435,1526231765,1526232666,1526233577,1526233705,1526234609,1526235518,1526235585,1526235635,1526235685,1526236590,1526236759,1526237661,1526238562,1526239467,1526240370,1526241281,1526242191,1526243093,1526243296,1526243693,1526244067,1526244377,1526244580,1526245485,1526245515,1526245565,1526245952,1526246418,1526246715,1526247379,1526248094,1526248261,1526248622,1526249651,1526250552,1526251462,1526252373,1526253275,1526254181,1526255088,1526256019,1526256043,1526256067,1526256976,1526257887,1526258789,1526259692,1526259971,1526260872,1526261774,1526262680,1526263591,1526277479,1526278048,1526278674,1526279052,1526279885,1526280200,1526281101,1526282004,1526282904,1526282965,1526283111,1526284020,1526284930,1526285838,1526286549,1526287454,1526287570,1526288480,1526289046,1526289472,1526290382,1526290666,1526291339,1526291506,1526291730,1526292026,1526292935,1526293735,1526294640,1526294695,1526294745,1526295651,1526295879,1526295955,1526296251,1526296340,1526296973,1526297582,1526297982,1526298887,1526299081,1526299105],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['trimming.npy'],
             'polarization_0': [1526200104,1526300014],
             'polarization_1': [1526200104,1526300014],
             'temperature': True
            },
        ],

        15263:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15263',
             'patches_subdirectory': '/marion2018/patches_100MHz/15263/1526300014',
             'ctimes': [1526300014,1526300835,1526301206,1526301632,1526301972,1526302216,1526303125,1526303371,1526304272,1526304756,1526304845,1526305750,1526306652,1526307489,1526307600,1526308501,1526309178,1526310079,1526310985,1526311892,1526312802,1526313709,1526314615,1526315521,1526316427,1526317338,1526318247,1526319159,1526320023,1526320216,1526321121,1526322023,1526322933,1526323720,1526324288,1526324662,1526324865,1526325161,1526326020,1526326929,1526327835,1526328743,1526328810,1526329720,1526330626,1526331274,1526331490,1526332392,1526332792,1526332855,1526332971,1526333483,1526333906,1526334332,1526335039,1526335947,1526336000,1526336539,1526336615,1526336900,1526337524,1526337802,1526338434,1526338704,1526339338,1526339614,1526340247,1526340521,1526341159,1526341422,1526342060,1526342329,1526342419,1526342963,1526343869,1526344774,1526345676,1526346583,1526347484,1526348244,1526349120,1526349296,1526349864,1526350764,1526351674,1526352582,1526353420,1526353626,1526354532,1526355441,1526356344,1526357250,1526357888,1526358177,1526363056,1526363975,1526364880,1526365282,1526366187,1526366230,1526367134,1526368037,1526368947,1526369849,1526370750,1526371657,1526372439,1526372629,1526373198,1526374103,1526374185,1526375095,1526375276,1526375300,1526376205,1526376988,1526377397,1526378298,1526378561,1526379174,1526380079,1526380407,1526380856,1526381438,1526381510,1526381655,1526381926,1526382280,1526382365,1526383274,1526383944,1526384283,1526384632,1526385540,1526386451,1526387353,1526388256],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['trimming.npy'],
             'polarization_0': [1526300030,1526388771],
             'polarization_1': [1526300030,1526388771],
             'temperature': False
            },
        ],

        15273:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15273',
             'patches_subdirectory': '/marion2018/patches_100MHz/15273/1527332452',
             'ctimes': [1527332452,1527333354,1527334260,1527335166,1527336077,1527336997,1527337393,1527337516,1527337878,1527338834,1527339735,1527340645,1527341546,1527342450,1527343351,1527344258,1527345164,1527346070,1527346976,1527347888,1527348789,1527349699,1527350601,1527351508,1527352419,1527353329,1527354240,1527355146,1527356056,1527356958,1527357869,1527358779,1527359690,1527360600,1527361511,1527362413,1527363319,1527364221,1527365123,1527366025,1527366936,1527367842,1527368744,1527369649,1527370561,1527371467,1527372372,1527373284,1527374186,1527375087,1527375998,1527376904,1527377810,1527378713,1527379220,1527379243,1527379267,1527380169,1527381074,1527381981,1527382886,1527383798,1527384699,1527385601,1527386508,1527387419,1527388324,1527389226,1527390137,1527391044,1527391945,1527392846,1527393748,1527394650,1527395561,1527396471,1527397374,1527398285,1527399195],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': [1527332463,1527400096],
             'polarization_1': [1527332463,1527400096],
             'temperature': True
            },
        ],

        15274:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15274',
             'patches_subdirectory': '/marion2018/patches_100MHz/15274/1527400097',
             'ctimes': [1527400097,1527400999,1527401900,1527402811,1527403721,1527404624,1527405530,1527406431,1527407339,1527408244,1527409146,1527411186,1527411262,1527412171,1527412214,1527413120,1527413355,1527418521,1527419431,1527420342,1527421248,1527423232,1527424142,1527424720,1527424792,1527425702,1527425908,1527426655,1527427561,1527427979,1527428752,1527429018,1527429919,1527430825,1527431732,1527432634,1527433539,1527434442,1527435352,1527436255,1527437165,1527437363,1527438200,1527438393,1527438417,1527438467,1527438517,1527438892,1527439796,1527439827,1527439937,1527440847,1527441083,1527441710,1527442611,1527443065,1527443192,1527443398,1527443487,1527444201,1527444277,1527444730,1527445104,1527445937,1527446841,1527447749,1527448048,1527448167,1527448373,1527448397,1527448663,1527449385,1527449608,1527449697,1527450059,1527450122,1527450925,1527451834,1527452744,1527453255,1527453388,1527454292,1527455001,1527455354,1527456260,1527456564,1527457473,1527458375,1527459283,1527460192,1527461099,1527462005,1527462908,1527463818,1527464720,1527465621,1527465645,1527465669,1527466255,1527467160,1527468063,1527468965,1527469867,1527470773,1527485129,1527485469,1527485502,1527485738,1527486255,1527486431,1527487000,1527487905,1527488324,1527489235,1527489795,1527490705,1527491606,1527492215,1527493130,1527498643,1527499545],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['trimming.npy'],
             'polarization_0': [1527400109,1527499840],
             'polarization_1': [1527400109,1527499840],
             'temperature': True
            },
        ],

        15277:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15277',
             'patches_subdirectory': '/marion2018/patches_100MHz/15277/1527724816',
             'ctimes': [1527724816,1527724840,1527724865,1527725109,1527728716,1527732319,1527735927,1527737188,1527740794,1527744402,1527748011,1527749655,1527750603,1527754215,1527757823,1527761435],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': [1527724872,1527768649],
             'polarization_1': [1527724872,1527768649],
             'temperature': True
            },

            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15277',
             'patches_subdirectory': '/marion2018/patches_100MHz/15277/1527768650',
             'ctimes': [1527768650,1527772262,1527774915,1527775919,1527777059,1527780666,1527784277,1527787890,1527789738,1527793343,1527796947],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': [1527768680,1527800560],
             'polarization_1': [1527768680,1527800560],
             'temperature': True
            },
        ],

        15278:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15278',
             'patches_subdirectory': '/marion2018/patches_100MHz/15278/1527800560',
             'ctimes': [1527800560,1527804168,1527805661,1527809263,1527811228,1527811251,1527829219,1527832370,1527832584,1527836191,1527837467,1527841070,1527841474,1527845086,1527847806,1527848689,1527851417,1527852297,1527855022,1527855904,1527858629,1527859512,1527860710,1527863115,1527863403,1527863694,1527867306,1527870909,1527874522,1527878129,1527881737,1527885338,1527888940,1527892544,1527893393,1527896996],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': [1527800590,1527897588],
             'polarization_1': [1527800590,1527897588],
             'temperature': True
            },
        ],

        15279:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15279',
             'patches_subdirectory': '/marion2018/patches_100MHz/15279/1527904939',
             'ctimes': [1527904939,1527908518,1527911379,1527912382,1527912750,1527913499,1527914678,1527916067,1527918286,1527919670,1527921820,1527925422,1527929034,1527929357,1527932964,1527936289,1527939892,1527943495,1527947099,1527950712,1527954324,1527957927,1527961527,1527961597,1527965208,1527968816,1527972424,1527973282,1527976632,1527980235,1527983842,1527984018,1527991461,1527995072,1527995736,1527997295,1527998057,1527998284],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': [1527904946,1527998648],
             'polarization_1': [1527904946,1527998648],
             'temperature': False
            },
        ],

        15280:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15280',
             'patches_subdirectory': '/marion2018/patches_100MHz/15280/1528036348',
             'ctimes': [1528036348,1528037011,1528037173,1528039305,1528042907,1528046516,1528050127,1528053730,1528057342,1528059083,1528060942,1528062695,1528064351,1528064549,1528064911,1528068153,1528068513,1528070433,1528071942,1528075548,1528077925,1528079160,1528081528,1528082769,1528084204,1528086376,1528087843,1528089988,1528091446,1528093592,1528095058,1528097195,1528098660],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': [1528036355,1528102264],
             'polarization_1': [1528036355,1528102264],
             'temperature': False
            },
        ],

        15281:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15281',
             'patches_subdirectory': '/marion2018/patches_100MHz/15281/1528100803',
             'ctimes': [1528100803,1528101929,1528102265,1528105872,1528109472,1528113076,1528116687,1528120295,1528123899,1528125672,1528126788,1528129284,1528129355,1528132891,1528136495,1528140098,1528141302,1528144909,1528146545,1528148270,1528151880,1528153469,1528164204,1528167815,1528168142,1528171744,1528172832,1528173102,1528176007,1528179609,1528183221,1528185831,1528186829,1528189442,1528190434,1528193047,1528196659],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': [1528100833,1528200265],
             'polarization_1': [1528100833,1528200265],
             'temperature': False
            },
        ],

        15283:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15283',
             'patches_subdirectory': '/marion2018/patches_100MHz/15283/1528302726',
             'ctimes': [1528302726,1528306334,1528309936,1528313540,1528317148,1528320760,1528324372,1528327984,1528329625,1528329649,1528340481,1528344084,1528347691,1528351295,1528351349,1528352284,1528352696,1528356299,1528357023,1528360630,1528364242,1528367846,1528371459,1528375069,1528378677,1528382281,1528385889,1528389490,1528392479,1528393091,1528396090,1528396696,1528399698,1528399925],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': [1528302750,1528403532],
             'polarization_1': [1528302750,1528403532],
             'temperature': False
            },
        ],

        15284:
        [
            {'component': '100MHz',
             'data_subdirectory': '/marion2018/data_100MHz/15284',
             'patches_subdirectory': '/marion2018/patches_100MHz/15284/1528400303',
             'ctimes': [1528400303,1528403532,1528407000,1528408826,1528409428,1528411285,1528413050,1528416032,1528416056,1528419667,1528423274,1528423510,1528426499,1528427543,1528431154,1528434758,1528438361,1528441969,1528445581,1528449189,1528452801,1528456413,1528460025,1528463633,1528467245,1528468069,1528468849,1528470363,1528473966,1528477573,1528479967,1528483574,1528487178,1528490789,1528492464,1528496072,1528497140,1528497864,1528498855],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['trimming.npy'],
             'polarization_0': [1528400332,1528502398],
             'polarization_1': [1528400332,1528502398],
             'temperature': False
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
             'polarization_0': [1524500618,1524600730],
             'polarization_1': [],
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
             'polarization_0': [1524600746,1524637710],
             'polarization_1': [],
             'temperature': True
            },

            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15246',
             'patches_subdirectory': '/marion2018/patches_70MHz/15246/1524658039',
             'ctimes': [1524658039,1524658940,1524659842,1524660743,1524661645,1524662547,1524663449,1524664351,1524665253,1524666155,1524667057,1524667959,1524668861,1524669763,1524670665,1524671567,1524672469,1524673375,1524674277,1524675179,1524676081,1524676983,1524677885,1524678787,1524679689,1524680590,1524681493,1524682394,1524683296,1524684198,1524685100,1524686002,1524686904,1524687806,1524688708,1524689610,1524690512,1524691414,1524692316,1524693218,1524694120,1524695022,1524695924,1524696825,1524697727,1524698629,1524699531],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_start.raw','float','time_sys_start.raw'),('time_stop.raw','float','time_sys_stop.raw')],
             'patch_files': [],
             'polarization_0': [],
             'polarization_1': [1524658043,1524700431],
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
             'polarization_0': [],
             'polarization_1': [1524700448,1524723375],
             'temperature': True
            },

            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15247',
             'patches_subdirectory': '/marion2018/patches_70MHz/15247/1524726386',
             'ctimes': [1524726386,1524727287,1524728194,1524729099,1524730006,1524730912,1524731818,1524732725,1524733631,1524734537,1524734592,1524735492,1524736399,1524737306,1524738211,1524739117,1524739721,1524740621,1524741529,1524742434,1524743336,1524744242,1524745148,1524746054,1524746961,1524747867,1524748773,1524749679,1524750586,1524751492,1524752398,1524753304,1524754211,1524755117,1524756023,1524756929,1524757836,1524758738,1524759645,1524760551,1524761456,1524762362,1524763269,1524764176,1524765081,1524765987,1524766893,1524767801,1524768706,1524769612,1524770519,1524771426,1524772331,1524773238,1524774143,1524775051,1524775956,1524776863,1524777768,1524778676,1524779581,1524780488,1524781393,1524782301,1524783206,1524784112,1524785018,1524785926,1524786831,1524787224,1524787247,1524794661,1524795562,1524796468,1524797374,1524798042,1524798197,1524798942,1524799098,1524799611,1524799848],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': [],
             'polarization_1': [1524726393,1524795685],
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
             'polarization_0': [],
             'polarization_1': [1524901435,1524934407],
             'temperature': True
            },
        ],

        15252:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15252',
             'patches_subdirectory': '/marion2018/patches_70MHz/15252/1525200400',
             'ctimes': [1525200400,1525200502,1525201014,1525201479,1525202384,1525202967,1525203082,1525203132,1525203670,1525204260,1525204470,1525205065,1525205657,1525205767,1525205882,1525206791,1525207692,1525208149,1525208554,1525208833,1525209506,1525209552,1525210017,1525210287,1525210739,1525210988,1525211047,1525211952,1525212608,1525213004,1525213590,1525214306,1525214538,1525214796,1525214937,1525215052,1525215167,1525215312,1525216217,1525217125,1525218026,1525218931,1525226540,1525227444,1525228351,1525229254,1525230122,1525230574,1525230702,1525230817,1525231726,1525232632,1525243112,1525243197,1525243614,1525244014,1525244920,1525245817,1525246071,1525246855,1525247113,1525247479,1525247819,1525248730,1525249018,1525249263,1525249629,1525249939,1525250848,1525251757,1525252248,1525253148,1525253247,1525253483,1525253572,1525254115,1525254419,1525254824,1525255007,1525255308,1525255553,1525256057,1525256202,1525256685,1525257336,1525257442,1525257758,1525257973,1525257997,1525258384,1525258663,1525259081,1525259127,1525259791,1525259867,1525260103,1525260667,1525261245,1525261342,1525261452,1525261658,1525261820,1525262446,1525263131,1525263328,1525263568,1525264469,1525264806,1525264882,1525265178,1525265393,1525265417,1525265606,1525266391,1525266593,1525267170,1525267579,1525267907,1525268052,1525268952,1525269480,1525269794,1525270134,1525270337,1525270608,1525271281,1525271724,1525271822,1525272138,1525272534,1525272567,1525272954,1525273609,1525273854,1525274022,1525274514,1525274736,1525275424,1525275582,1525276326,1525276491,1525277236,1525277401,1525278006,1525278302,1525278631,1525278828,1525279205,1525279728,1525280112,1525280629,1525281015,1525281536,1525281916,1525282437,1525282821,1525283339,1525283556,1525283848,1525284093,1525284246,1525284997,1525285152,1525286033,1525286059,1525286585,1525286964,1525287055,1525287429,1525287643,1525287867,1525288021,1525288774,1525288867,1525289194,1525289438,1525289672,1525290346,1525290377,1525290595,1525290727,1525291084,1525291177,1525292081,1525292985,1525293886,1525294791,1525294822,1525295731,1525296015,1525296177,1525296780,1525297647,1525297853,1525298075,1525298207,1525298322,1525298804,1525298872,1525299278,1525299830],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
             'polarization_0': [],
             'polarization_1': [1525200408, 1525300214],
             'temperature': False
            },
        ],

#        15277:
#        [
#            {'component': '70MHz',
#             'data_subdirectory': '/marion2018/data_70MHz/15277',
#             'patches_subdirectory': '/marion2018/patches_70MHz/15277/1527700219',
#             'ctimes': [1527700219,1527701130,1527701160,1527702031,1527704763,1527708375,1527711987,1527712336,1527713246,1527714152,1527715058,1527715323,1527715969,1527716275,1527716871,1527717461,1527720243,1527723846,1527724820,1527724843,1527735677,1527739288,1527742897,1527743705,1527743858,1527745255,1527748867,1527752473,1527756087,1527759693,1527763307,1527766913,1527770517,1527774124,1527777727,1527781340,1527784518,1527784952,1527787362,1527788121,1527790964,1527791723,1527792004,1527794572,1527795607,1527799218],
#             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
#             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
#             'patch_files': ['reordering.npy','offsetting.npy','trimming.npy'],
#             'polarization_0': [],
#             'polarization_1': [],
#             'temperature': False
#            },
#        ],

        15281:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15281',
             'patches_subdirectory': '/marion2018/patches_70MHz/15281/1528101778',
             'ctimes': [1528101778,1528101819,1528105425,1528109034,1528111192,1528114800,1528116531,1528119764,1528120398,1528120706,1528121489,1528122415,1528126026,1528129638,1528133250,1528134251,1528137855,1528139351,1528142958,1528146566,1528150170,1528153773,1528155771,1528156827,1528156851,1528160454,1528162794],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': [1528101778,1528162794],
             'polarization_1': [],
             'temperature': False
            },
        ],

        15285:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15285',
             'patches_subdirectory': '/marion2018/patches_70MHz/15285/1528501039',
             'ctimes': [1528501039,1528501425,1528502425,1528502449,1528502472,1528505037,1528506075,1528508645,1528509682,1528512249,1528512969,1528515861,1528516576,1528519468,1528520179,1528523076,1528523787,1528526684,1528526764,1528527398,1528530375,1528531011,1528533557,1528534623,1528538234,1528541838,1528545443,1528549050,1528552662,1528556458,1528557444,1528558626,1528561036,1528564647,1528568251,1528571863,1528575471,1528579004,1528580527,1528584138,1528587742,1528588824,1528588847,1528588871,1528592482],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': [1528501039,1528592482],
             'polarization_1': [],
             'temperature': False
            },
        ],

        15302:
        [
            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15302',
             'patches_subdirectory': '/marion2018/patches_70MHz/15302/1530293227',
             'ctimes': [1530293227],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': [1530293236,1530293732],
             'polarization_1': [],
             'temperature': False
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
             'patch_files': ['reordering.npy','trimming.npy'],
             'polarization_0': [1535726620,1535800071],
             'polarization_1': [],
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
             'polarization_0': [1536201218,1536225832],
             'polarization_1': [],
             'temperature': True
            },

            {'component': '70MHz',
             'data_subdirectory': '/marion2018/data_70MHz/15362',
             'patches_subdirectory': '/marion2018/patches_70MHz/15362/1536233063',
             'ctimes': [1536233063,1536234172,1536234844,1536235374,1536235929,1536244041,1536247648,1536249622,1536249988,1536250539,1536251099,1536251459,1536251621,1536252013,1536252503,1536252968,1536253264,1536253426,1536254366,1536256810,1536257072,1536257606,1536259437,1536259923,1536260906,1536261237,1536261741,1536264214,1536266625,1536267193,1536267658,1536269335,1536270120,1536272616,1536272726,1536273420,1536273604,1536274199,1536275987,1536276534,1536277245,1536277610,1536278418,1536278442,1536278465,1536281514,1536289794,1536289895,1536291259,1536292005,1536292388,1536293205,1536293682,1536293957,1536294323,1536294386,1536295801,1536295916,1536295966,1536296106,1536296402,1536297056],
             'scio_files': [('pol0.scio','pol0.scio'),('pol1.scio','pol1.scio'),('cross_real.scio','cross_real.scio'),('cross_imag.scio','cross_imag.scio')],
             'raw_files': [('acc_cnt1.raw','int32','acc_cnt1.raw'),('acc_cnt2.raw','int32','acc_cnt2.raw'),('fft_of_cnt.raw','int32','fft_of_cnt.raw'),('fft_shift.raw','int64','fft_shift.raw'),('fpga_temp.raw','float','fpga_temp.raw'),('pi_temp.raw','int32','pi_temp.raw'),('sync_cnt1.raw','int32','sync_cnt1.raw'),('sync_cnt2.raw','int32','sync_cnt2.raw'),('sys_clk1.raw','int32','sys_clk1.raw'),('sys_clk2.raw','int32','sys_clk2.raw'),('time_sys_start.raw','float','time_sys_start.raw'),('time_sys_stop.raw','float','time_sys_stop.raw'),('time_rtc_start.raw','float','time_rtc_start.raw'),('time_rtc_stop.raw','float','time_rtc_stop.raw')],
             'patch_files': [],
             'polarization_0': [1536233071,1536299981],
             'polarization_1': [],
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
             'polarization_0': [1537108262,1537203116],
             'polarization_1': [],
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
             'polarization_0': [1538404724,1538501452],
             'polarization_1': [],
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
             'patch_files': ['reordering.npy','offsetting.npy'],
             'polarization_0': [1539527463,1539601140],
             'polarization_1': [],
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
             'polarization_0': [1540480560,1540500067],
             'polarization_1': [],
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
             'patch_files': ['offsetting.npy'],
             'polarization_0': [1541949315,1541988914],
             'polarization_1': [],
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
             'polarization_0': [1542896216,1542902681],
             'polarization_1': [],
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
             'polarization_0': [1544366104,1544399983],
             'polarization_1': [],
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
             'polarization_0': [1545321498,1545401552],
             'polarization_1': [],
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
             'polarization_0': [1546960561,1546996266],
             'polarization_1': [],
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
             'polarization_0': [1548512473,1548601493],
             'polarization_1': [],
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
             'polarization_0': [1549812847,1549900769],
             'polarization_1': [],
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
             'polarization_0': [1550503735,1550601920],
             'polarization_1': [],
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
             'polarization_0': [1552223461,1552300635],
             'polarization_1': [],
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
             'polarization_0': [1554041608,1554101779],
             'polarization_1': [],
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
             'polarization_0': [1555335080,1555400284],
             'polarization_1': [],
             'temperature': True
            },
        ],
    },

    'switch':
    {
        15244:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15244',
             'patches_subdirectory': '',
             'ctimes': [1524485396],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15244',
             'patches_subdirectory': '',
             'ctimes': [1524485398],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15244',
             'patches_subdirectory': '',
             'ctimes': [1524487505],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15244',
             'patches_subdirectory': '',
             'ctimes': [1524487507],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15244',
             'patches_subdirectory': '',
             'ctimes': [1524494732],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_pi.raw','float','time_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15244',
             'patches_subdirectory': '',
             'ctimes': [1524495095],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

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

        15252:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525205510],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525213197],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525220883],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525228570],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525236257],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525243943],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525251630],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525259317],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525267003],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525274690],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525282377],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525290063],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15252',
             'patches_subdirectory': '',
             'ctimes': [1525297750],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15260:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526015423],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526015425],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526023109],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526030796],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526046752],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526046754],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526053972],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526054438],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526061213],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526062125],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526068456],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526069812],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526075697],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526077498],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526082958],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526085185],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526090231],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526092872],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15260',
             'patches_subdirectory': '',
             'ctimes': [1526097475],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },
        ],

        15261:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526100558],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526104719],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526108245],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526111959],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526115932],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526119198],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526123618],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526126439],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526131305],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526133678],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526138992],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526140928],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526146678],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526148170],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526154365],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526155414],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526162052],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526162658],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526169738],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526169904],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526177146],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526177425],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526184396],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526185112],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526191642],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526192798],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15261',
             'patches_subdirectory': '',
             'ctimes': [1526198886],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },
        ],

        15262:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526200485],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526212229],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526212231],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526219473],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526219916],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526226716],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526227602],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526233955],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526235289],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526241194],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526242976],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526248431],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526250662],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526255671],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526258349],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526266036],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526273722],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526281409],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526289096],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15262',
             'patches_subdirectory': '',
             'ctimes': [1526296782],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15263:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526304469],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526312156],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526319842],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526327529],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526335216],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526342902],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526350589],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526358276],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526365962],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526373649],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526381336],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526389022],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15263',
             'patches_subdirectory': '',
             'ctimes': [1526396709],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15273:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527332439],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527332441],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527339684],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527340126],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527346928],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527347812],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527354171],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527355499],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527361417],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527363186],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527368659],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527370872],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527375899],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527378559],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527386245],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15273',
             'patches_subdirectory': '',
             'ctimes': [1527393932],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15274:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527401619],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527409306],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527411173],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527411175],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527418510],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527418512],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527419019],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527419021],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527423221],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527423223],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527424120],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527424122],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527431338],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527432123],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527438554],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527440126],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527445768],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527448130],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527452983],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527456133],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527460192],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527464136],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527472139],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527480142],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527488144],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527498632],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15274',
             'patches_subdirectory': '',
             'ctimes': [1527498634],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },
        ],

        15277:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527701917],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527704262],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527709143],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527712265],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527716374],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527720268],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527723607],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527728271],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527736274],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527744277],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527752279],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527760282],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527765511],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527765513],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527765685],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527765687],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527766641],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527766643],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527766859],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527766861],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527767380],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527767382],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527774633],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527775384],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527781878],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527783387],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527789125],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527791390],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527796369],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15277',
             'patches_subdirectory': '',
             'ctimes': [1527799394],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15278:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527803614],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527807397],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527810859],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527815400],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527823403],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527831407],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527839410],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527847413],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527855417],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527863420],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527871423],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527879427],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527887430],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15278',
             'patches_subdirectory': '',
             'ctimes': [1527895433],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15279:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527903437],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527911440],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527919443],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527927447],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527935450],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527943453],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527951457],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527959460],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527967463],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527975467],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527983470],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527991473],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15279',
             'patches_subdirectory': '',
             'ctimes': [1527999477],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15280:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528007480],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528015483],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528023486],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528031490],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528039493],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528047497],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528055500],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528063503],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528071507],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528079510],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528087513],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15280',
             'patches_subdirectory': '',
             'ctimes': [1528095517],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15281:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528103520],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528111523],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528119527],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528127530],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528135533],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528143537],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528151540],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528159543],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528167547],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528175550],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528183553],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528191557],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15281',
             'patches_subdirectory': '',
             'ctimes': [1528199560],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15283:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528303603],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528311607],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528319610],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528327613],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528335617],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528343620],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528351623],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528359627],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528367630],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528375633],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528383637],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528391640],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15283',
             'patches_subdirectory': '',
             'ctimes': [1528399643],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15284:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528407647],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528415650],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528423653],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528431657],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528439660],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528447663],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528455667],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528463670],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528471673],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528479677],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528487680],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15284',
             'patches_subdirectory': '',
             'ctimes': [1528495683],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },
        ],

        15302:
        [
            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530206171],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530214174],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530222177],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530230180],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530238183],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530246186],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530254189],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530262191],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530278305],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530278307],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530285547],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530286308],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530292789],
             'scio_files': [],
             'raw_files': [('temp_100A_bot_lna.raw','float','temp_100A_bot_lna.raw'),('temp_100A_noise.raw','float','temp_100A_noise.raw'),('temp_100A_switch.raw','float','temp_100A_switch.raw'),('temp_100A_top_lna.raw','float','temp_100A_top_lna.raw'),('temp_100B_bot_lna.raw','float','temp_100B_bot_lna.raw'),('temp_100B_noise.raw','float','temp_100B_noise.raw'),('temp_100B_switch.raw','float','temp_100B_switch.raw'),('temp_100B_top_lna.raw','float','temp_100B_top_lna.raw'),('temp_100_ambient.raw','float','temp_100_ambient.raw'),('temp_70A_bot_lna.raw','float','temp_70A_bot_lna.raw'),('temp_70A_noise.raw','float','temp_70A_noise.raw'),('temp_70A_switch.raw','float','temp_70A_switch.raw'),('temp_70A_top_lna.raw','float','temp_70A_top_lna.raw'),('temp_70B_bot_lna.raw','float','temp_70B_bot_lna.raw'),('temp_70B_noise.raw','float','temp_70B_noise.raw'),('temp_70B_switch.raw','float','temp_70B_switch.raw'),('temp_70B_top_lna.raw','float','temp_70B_top_lna.raw'),('temp_70_ambient.raw','float','temp_70_ambient.raw'),('temp_pi.raw','float','temp_pi.raw'),('temp_snapbox.raw','float','temp_snapbox.raw'),('time_rtc_pi.raw','float','time_rtc_pi.raw'),('time_start_therms.raw','float','time_start_therms.raw'),('time_stop_therms.raw','float','time_stop_therms.raw'),('time_sys_pi.raw','float','time_sys_pi.raw')],
             'patch_files': [],
            },

            {'component': 'switch',
             'data_subdirectory': '/marion2018/switch_data/15302',
             'patches_subdirectory': '',
             'ctimes': [1530294312],
             'scio_files': [('antenna.scio','antenna.scio'),('noise.scio','noise.scio'),('open.scio','open.scio'),('res100.scio','res100.scio'),('res50.scio','res50.scio'),('short.scio','short.scio')],
             'raw_files': [],
             'patch_files': [],
            },

        ]

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
