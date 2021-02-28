# NumPy
import numpy as np

# Prizmatoid
import prizmatoid as pzt

# Matplotlib
from matplotlib import pyplot as plt

# Other Useful Libraries
from copy import deepcopy
from scipy.optimize import minimize
from itertools import permutations


def plot_timestamps(prizm_data, antenna, display_slices=[slice(0, -1, None)]):
    """ Plots the data timestamps. """

    # Plots the timestamps.
    plt.figure(1, figsize=(10,10))
    for entry in display_slices:
        plt.plot(np.arange(0, len(prizm_data[antenna]['time_sys_start.raw']))[entry],
                 prizm_data[antenna]['time_sys_start.raw'][entry])

    return


def plot_waterfall(prizm_data, antenna, pol, display_slice=slice(0, -1, None), vmin=4.5, vmax=8.5):
    """ Plots the data timestamps. """

    # Plots the timestamps.
    pzt.add_switch_flags(prizm_data, antennas=[antenna])

    # Stores the different flags into NumPy arrays.
    select_antenna = (pzt.shrink_flag(prizm_data[antenna]['switch_flags']['antenna.scio'], (1,1)) == 1)
    select_res100 = (pzt.shrink_flag(prizm_data[antenna]['switch_flags']['res100.scio'], (1,1)) == 1)
    select_res50 = (pzt.shrink_flag(prizm_data[antenna]['switch_flags']['res50.scio'], (1,1)) == 1)
    select_short = (pzt.shrink_flag(prizm_data[antenna]['switch_flags']['short.scio'], (1,1)) == 1)
    select_noise = (pzt.shrink_flag(prizm_data[antenna]['switch_flags']['noise.scio'], (1,1)) == 1)

    # Creates filters for the slice of interest.
    f_antenna = np.concatenate(( np.zeros_like(select_antenna[slice(0, display_slice.start, None)]),
                                select_antenna[display_slice],
                                np.zeros_like(select_antenna[slice(display_slice.stop, len(select_antenna), None)])
                                ))

    f_res100 = np.concatenate(( np.zeros_like(select_res100[slice(0, display_slice.start, None)]),
                                select_res100[display_slice],
                                np.zeros_like(select_res100[slice(display_slice.stop, len(select_res100), None)])
                                ))

    f_res50 = np.concatenate(( np.zeros_like(select_res50[slice(0, display_slice.start, None)]),
                               select_res50[display_slice],
                               np.zeros_like(select_res50[slice(display_slice.stop, len(select_res50), None)])
                               ))

    f_short = np.concatenate(( np.zeros_like(select_short[slice(0, display_slice.start, None)]),
                               select_short[display_slice],
                               np.zeros_like(select_short[slice(display_slice.stop, len(select_short), None)])
                               ))

    f_noise = np.concatenate(( np.zeros_like(select_noise[slice(0, display_slice.start, None)]),
                               select_noise[display_slice],
                               np.zeros_like(select_noise[slice(display_slice.stop, len(select_noise), None)])
                               ))

    # Selects the data from the input polarization `pol` channel associated with the different observing modes.
    antenna_data = prizm_data[antenna][pol + '.scio'][f_antenna]
    res100_data = prizm_data[antenna][pol + '.scio'][f_res100]
    res50_data = prizm_data[antenna][pol + '.scio'][f_res50]
    short_data = prizm_data[antenna][pol + '.scio'][f_short]
    noise_data = prizm_data[antenna][pol + '.scio'][f_noise]

    # Plots the 'antenna.scio' mode for the polarization channel of interest on a logarithmic color scale.
    plt.figure(1, figsize=(10, 10))
    plt.title(pol +  ' (Antenna)')
    plt.imshow(np.log10(antenna_data), vmin=vmin, vmax=vmax)

    # Plots the 'short.scio' mode for the polarization channel of interest on a logarithmic color scale.
    plt.figure(2, figsize=(20, 10))
    plt.title(pol + ' (Short)')
    plt.imshow(np.log10(short_data), vmin=vmin, vmax=vmax)

    # Plots the 'res50.scio' mode for the polarization channel of interest on a logarithmic color scale.
    plt.figure(3, figsize=(20, 10))
    plt.title(pol +  ' (50 Ohm)')
    plt.imshow(np.log10(res50_data), vmin=vmin, vmax=vmax)

    # Plots the 'res100.scio' mode for the polarization channel of interest on a logarithmic color scale.
    plt.figure(4, figsize=(20, 10))
    plt.title(pol +  ' (100 Ohm)')
    plt.imshow(np.log10(res100_data), vmin=vmin, vmax=vmax)

    # Plots the 'noise.scio' mode for the polarization channel of interest on a logarithmic color scale.
    plt.figure(5, figsize=(20, 10))
    plt.title(pol +  ' (Noise)')
    plt.imshow(np.log10(noise_data), vmin=vmin, vmax=vmax)

    return


def find_jumps(time_data, time_scale):
    """ Finds jumps in the time-ordered data. """
    
    # Obtains the time increments and decrements of `time_data`.
    increments = np.diff(time_data)
    
    # Obtains the jump `magnitudes`.
    magnitudes = increments[np.abs(increments) > time_scale]
    magnitudes = np.append(0, magnitudes)
    
    # Finds where the `timestamps` change by more than the reference `timescale`.
    jumps = np.where(np.abs(increments) > time_scale)[0] + 1
    
    # Appends the 0-th index to `jumps`.
    jumps = np.append(0, jumps)
    
    # Checks whether a jumps occurs on the last entry of `timestamps`. If not,
    # the index associated with that entry is appended to `jumps`.
    if len(time_data) - 1 not in jumps:
        jumps = np.append(jumps, len(time_data))
    
    # Creates a list of slices delimited by the entries of `jumps`.
    slices = []
    for start, end in zip(jumps[:-1], jumps[1:]):
        slices.append(slice(start, end, None))
    
    # Return `slices` and `magninutes`.
    return slices, magnitudes


def find_prizm_jumps(prizm_data, antenna, f):
    """ Finds jumps in the timestamp. """

    # Gets the timestamp information stored in `prizm_data`.
    time_data = prizm_data[antenna]['time_sys_start.raw']

    # Finds the reference time scale with respect to which jumps will be
    # identified.
    time_scale = f*np.min(np.diff(prizm_data['switch']['antenna.scio'][:,1]))

    # Obtains the time `slices` and `magnitudes` associated with the jumps in
    # the data.
    slices, magnitudes = find_jumps(time_data, time_scale)

    # Return `slices`.
    return slices, magnitudes


def remove_glitches(prizm_data, antenna, slices, magnitudes):
    """ Removes jumps which caused timestamps to go back in time. """

    # Gets the timestamp information stored in `prizm_data`.
    timestamps_start = prizm_data[antenna]['time_sys_start.raw']
    timestamps_stop = prizm_data[antenna]['time_sys_stop.raw']

    # Sets all non-negative entries of `magnitudes` to zero.
    negative_magnitudes = np.array([-entry if entry < 0
                                    else 0
                                    for entry in magnitudes])

    # Applies each offset to their respective slice.
    for i, slice in enumerate(slices):
        if negative_magnitudes[i] != 0:
            timestamps_start[slice] += negative_magnitudes[:i+1].sum()
            timestamps_stop[slice] += negative_magnitudes[:i+1].sum()
        else:
            continue

    # Stores the modified `timestamps` in `data`.
    prizm_data[antenna]['time_sys_start.raw'] = timestamps_start
    prizm_data[antenna]['time_sys_stop.raw'] = timestamps_stop

    return


def apply_offset(prizm_data, antenna, slices, offsets):
    """ Apply a different offset to each slice of PRIZM's timestamp data. """

    # Copies the input dictionary.
    data = deepcopy(prizm_data)

    # Gets the timestamp information stored in `prizm_data`.
    timestamps_start = data[antenna]['time_sys_start.raw']
    timestamps_stop = data[antenna]['time_sys_stop.raw']

    # Applies each offset to their respective slice.
    for slice, offset in zip(slices, offsets):
        timestamps_start[slice] += offset
        timestamps_stop[slice] += offset

    # Stores the modified `timestamps` in `data`.
    data[antenna]['time_sys_start.raw'] = timestamps_start
    data[antenna]['time_sys_stop.raw'] = timestamps_stop

    # Returns a modified data dictionary.
    return data


def find_outliers(prizm_data, antenna, pol_channel):
    """ Counts how many outliers exist for each observation mode. """

    # Adjusts thresholds accordinh to the `antenna` under consideration.
    if antenna == '100MHz':
        # `noise.scio`.
        noise_upper_threshold = 8.4

        # `res50.scio`
        res50_lower_threshold = 7.96
        res50_upper_threshold = 8.0

        # `res100.scio`
        res100_lower_threshold = 7.94
        res100_upper_threshold = 7.97

        # `noise.scio`.
        short_lower_threshold = 7.52

    if antenna == '70MHz':
        # `noise.scio`.
        noise_upper_threshold = 8.8

        # `res50.scio`
        res50_lower_threshold = 7.94
        res50_upper_threshold = 7.96

        # `res100.scio`
        res100_lower_threshold = 7.91
        res100_upper_threshold = 7.93

        # `noise.scio`.
        short_lower_threshold = 7.5

    # Creates filters which select all possible observation modes, i.e.,
    # `antenna.scio`, `res50.scio`, `res100.scio`, `short.scio`, and
    # `noise.scio`.
    select_antenna = (prizm_data[antenna]['switch_flags']['antenna.scio'] == 1)
    select_res100 = (prizm_data[antenna]['switch_flags']['res100.scio'] == 1)
    select_res50 = (prizm_data[antenna]['switch_flags']['res50.scio'] == 1)
    select_short = (prizm_data[antenna]['switch_flags']['short.scio'] == 1)
    select_noise = (prizm_data[antenna]['switch_flags']['noise.scio'] == 1)

    # Selects the data from the `pol_channel` associated with the
    # different observing modes.
    antenna_data_pol = prizm_data[antenna][pol_channel][select_antenna]
    res100_data_pol = prizm_data[antenna][pol_channel][select_res100]
    res50_data_pol = prizm_data[antenna][pol_channel][select_res50]
    short_data_pol = prizm_data[antenna][pol_channel][select_short]
    noise_data_pol = prizm_data[antenna][pol_channel][select_noise]

    # Identifies the relevant outliers.
    noise_outliers = np.sum(np.log10(noise_data_pol[:,3150]) < noise_upper_threshold)
    
    res50_outliers = np.sum(np.logical_and(
                            np.log10(res50_data_pol[:,1200]) > res50_upper_threshold,
                            np.log10(res50_data_pol[:,1200]) < res50_lower_threshold
                            ))

    res100_outliers = np.sum(np.logical_and(
                             np.log10(res100_data_pol[:,1200]) > res100_upper_threshold,
                             np.log10(res100_data_pol[:,1200]) < res50_lower_threshold
                             ))

    short_outliers = np.sum(np.log10(short_data_pol[:,1200]) > short_lower_threshold)

    # Returns the number of ouliers obtained above as an array.
    outliers = np.array([noise_outliers,
                         res50_outliers,
                         res100_outliers,
                         short_outliers])

    return outliers


def outlier_count(offsets, prizm_data, antenna, pol_channel, slices):
    """ Applies `offsets`, updates the switch flags, and returns outliers counts. """

    # Applies the absolute value of `offsets`.
    data = apply_offset(prizm_data, antenna=antenna, slices=slices, offsets=offsets)

    # Updates the switch flags.
    pzt.add_switch_flags(data, antennas=[antenna])

    # Computes the outliers given the new flags.
    count = np.sum(find_outliers(data, antenna, pol_channel))

    # Returns the new outlier counts.
    return count


def minimize_outliers(prizm_data, antenna, pol_channel, slices, slice_nmbr):
    """ Finds the offset which ,inimizes the number of outliers for a given timestamp slice. """

    # Defines the `initial_steps` with respect to which the outlier curve will be sampled at first.
    initial_step = np.min(np.abs(np.diff(prizm_data['switch']['res50.scio'][:,1])))

    # Defines the `final_steps` with respect to which the outlier curve will be sampled last.
    timestamp_increments = np.abs(np.diff(prizm_data[antenna]['time_sys_start.raw']))
    final_step = np.min(timestamp_increments[timestamp_increments != 0])

    # Defines the `bound` within which to look for a minimum in the outlier curve.
    bound = np.max(np.diff(prizm_data['switch']['res50.scio'][:,1]))

    # Generates the initial `search_grid` and initializes the `outlier_curve` array.
    #search_grid = np.arange(-bound, bound, initial_step/2)
    search_grid = np.arange(0, 1.5*bound, initial_step/2)
    search_grid = np.concatenate((np.linspace(-initial_step, 0, 100), search_grid))
    outlier_curve = np.zeros_like(search_grid)

    print("First Loop.")
    # Finds the outlier minimum.
    for j, grid_point in enumerate(search_grid):
        offsets = np.zeros_like(slices)
        offsets[slice_nmbr] = grid_point
        outlier_curve[j] = outlier_count(offsets, prizm_data, antenna, pol_channel, slices=slices)

    # Identifies the `minima` of `outlier_curve` and its respectice `point`.
    minima = np.argmin(outlier_curve)
    point = search_grid[minima]

    # Refines the `search_grid` and the `outlier_curve`.
    search_grid = np.arange(point - initial_step/2, point + initial_step/2, final_step/2)
    outlier_curve = np.zeros_like(search_grid)

    print("Second Loop.")
    # Finds the outlier minimum.
    for j, grid_point in enumerate(search_grid):
        offsets = np.zeros_like(slices)
        offsets[slice_nmbr] = grid_point
        outlier_curve[j] = outlier_count(offsets, prizm_data, antenna, pol_channel, slices=slices)

    # Identifies the `minima` of `outlier_curve` and their respectice grid `points`.
    minima = np.where(outlier_curve == outlier_curve.min())[0]
    points = search_grid[minima]

    # returns the `minima`.
    return points.mean()


def tetris(prizm_data, antenna, f=0.25):

    # Initializes the `control` variable which controls the 'while' loop below.
    control = True

    # Finds the reference time scale with respect to which jumps will be
    # identified.
    time_scale = f*np.min(np.diff(prizm_data['switch']['antenna.scio'][:,1]))

    # Creates `indices` which will keep a record of how the data indices have
    # been reshuffled.
    indices = np.arange(0, len(prizm_data[antenna]['time_sys_start.raw']))

    # Loads the 'time_sys_start.raw' entry of `prizm_data`.
    timestamps = prizm_data[antenna]['time_sys_start.raw']

    # Finds the `slices` in which the data is split, as well as their relative
    # `offsets`.
    slices, offsets = find_prizm_jumps(prizm_data, antenna, f)

    # Loops while `control` is `True`.
    while control:

        # Stops the 'while' loop if no slices permutations can be generated.
        if len(slices) - 1 == 2:
            control = False

        # Loops over all permutations of `slices` in an attempt to find a
        # configuration for which the data is segmented in fewer parts.
        for permutation in permutations(range(1, len(slices) - 1), 2):

            # Initializes `permutation_tuple` and performs the pair permutation
            # encapsulated in `permutation`.
            permutation_tuple = np.arange(0, len(slices))
            permutation_tuple[permutation[0]] = permutation[1]
            permutation_tuple[permutation[1]] = permutation[0]

            # Checks whether the last permutation has been reached. If so,
            # `control` is set to `False`.
            if permutation[0] == len(slices) - 2 and permutation[1] == len(slices) - 3:
                control = False

            # Initializes a copy of `indices` and `timestamps` for reshuffling.
            indices_copy = indices[slices[permutation_tuple[0]]]
            timestamps_copy = timestamps[slices[permutation_tuple[0]]]

            for entry in permutation_tuple[1:]:
                    timestamps_copy = np.concatenate((timestamps_copy, timestamps[slices[entry]]))
                    indices_copy = np.concatenate((indices_copy, indices[slices[entry]]))

            # Finds the number of segments into which the `timestamps` data is
            # split after the slices permutations have been applied.
            new_slices, new_offsets = find_jumps(timestamps_copy, time_scale)

            # Records the permutation which results in the smallest number of
            # segements in `timestamps`.
            if len(new_slices) < len(slices):
                # Re-defines `slices` and `offsets` after a successful reshuffling.
                slices = deepcopy(new_slices)
                offsets = deepcopy(new_offsets)

                # Updates `indices` and `timestamps` after reshuffling.
                indices = deepcopy(indices_copy)
                timestamps = deepcopy(timestamps_copy)

                # Exits 'for' loop.
                break

    return indices


def cut(prizm_data, antenna, slices):

    

    # Cuts the data which falls within the input list of `slices`.
    prizm_data[antenna]['pol0.scio'] = np.delete(prizm_data[antenna]['pol0.scio'], np.r_[tuple(slices)], axis=0)
    prizm_data[antenna]['pol1.scio'] = np.delete(prizm_data[antenna]['pol1.scio'], np.r_[tuple(slices)], axis=0)
    prizm_data[antenna]['time_sys_start.raw'] = np.delete(prizm_data[antenna]['time_sys_start.raw'], np.r_[tuple(slices)], axis=0)
    prizm_data[antenna]['time_sys_stop.raw'] = np.delete(prizm_data[antenna]['time_sys_stop.raw'], np.r_[tuple(slices)], axis=0)

    return


def reshuffle(prizm_data, antenna, indices):

    # Substitutes the information in `prizm_data` by its reshuffled version, as
    # described the input list of `indices`.
    prizm_data[antenna]['pol0.scio'] = prizm_data[antenna]['pol0.scio'][indices]
    prizm_data[antenna]['pol1.scio'] = prizm_data[antenna]['pol1.scio'][indices]
    prizm_data[antenna]['time_sys_start.raw'] = prizm_data[antenna]['time_sys_start.raw'][indices]
    prizm_data[antenna]['time_sys_stop.raw'] = prizm_data[antenna]['time_sys_stop.raw'][indices]

    return


def swap(prizm_data, antenna, slices, swap_slices, indices=np.array([0])):

    if len(indices) < len(prizm_data[antenna]['time_sys_start.raw']):
        # Creates `indices` which will keep a record of how the data indices have
        # been reshuffled.
        indices = np.arange(0, len(prizm_data[antenna]['time_sys_start.raw']))

    # Initializes `permutation_tuple` and performs the pair permutation
    # encapsulated in `swap_slices`.
    permutation_tuple = np.arange(0, len(slices))
    permutation_tuple[swap_slices[0]] = swap_slices[1]
    permutation_tuple[swap_slices[1]] = swap_slices[0]

    # Initializes a copy of `indices` and `timestamps` for reshuffling.
    swapped_indices = indices[slices[permutation_tuple[0]]]

    for entry in permutation_tuple[1:]:
            swapped_indices = np.concatenate((swapped_indices, indices[slices[entry]]))

    # Substitutes the information in `prizm_data` by its reshuffled version, as
    # described the input list of `indices`.
    reshuffle(prizm_data, antenna, swapped_indices)

    # Swap elements in `slices`.
    temporary_slice = slices[swap_slices[1]]
    slices[swap_slices[1]] = slices[swap_slices[0]]
    slices[swap_slices[0]] = temporary_slice

    # Returns the reshuffled `indices`.
    return swapped_indices


def shift(prizm_data, antenna, slices, shift_slice, to_slice_position, i=np.array([0])):

    # Creates `indices` which will keep a record of how the data indices have
    # been reshuffled.
    indices = np.arange(0, len(prizm_data[antenna]['time_sys_start.raw']))

    if len(i) < len(prizm_data[antenna]['time_sys_start.raw']):
        i = indices

    # Initializes a copy of `indices` and `timestamps` for reshuffling.
    if to_slice_position == 0:
        swapped_indices = indices[slices[shift_slice]]
        swapped_indices = np.concatenate((swapped_indices, indices[slices[0]]))
        swapped_slices = np.array([slices[shift_slice]])
        swapped_slices = np.concatenate((swapped_slices, np.array([slices[0]])))

    else:
        swapped_indices = indices[slices[0]]
        swapped_slices = np.array([slices[0]])

    # Performs the shift of both `indices` and `slices`.
    for entry in np.arange(1, len(slices)):
        if entry == shift_slice:
            continue;

        elif entry == to_slice_position:
            swapped_indices = np.concatenate((swapped_indices, indices[slices[shift_slice]]))
            swapped_indices = np.concatenate((swapped_indices, indices[slices[entry]]))
            swapped_slices = np.concatenate((swapped_slices, np.array([slices[shift_slice]])))
            swapped_slices = np.concatenate((swapped_slices, np.array([slices[entry]])))

        else:
            swapped_indices = np.concatenate((swapped_indices, indices[slices[entry]]))
            swapped_slices = np.concatenate((swapped_slices, np.array([slices[entry]])))

    # Substitutes the information in `prizm_data` by its reshuffled version, as
    # described the input list of `indices`.
    reshuffle(prizm_data, antenna, swapped_indices)

    # Updates the order of the input `slices`.
    slices = swapped_slices.tolist()

    # Returns the reshuffled `indices`.
    return i[swapped_indices]


def fix_timestamps(prizm_data, antenna, pol_channel, slices, target_slices, initial_offsets):
    """ Attemps to fix the timestamps. """

    # Initializes the array which will store the final offsets.
    result = np.zeros_like(slices)

    # Loops over each slice, mnimizing one at a time.
    for slice_number in target_slices:

        print("Slice: ", slice_number, ".")
        # Minimizes the 'outlier_count' function.
        result[slice_number] = minimize_outliers(prizm_data, antenna, pol_channel, slices, slice_number)

    # Returns `offsets`.
    return result
