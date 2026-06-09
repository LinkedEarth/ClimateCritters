import os
import csv
import re
import lipd as lpd
import numpy as np
import pyleoclim as pyleo
import plotly.express as px

from IPython.display import clear_output

def _get_tso_indices(tso,val_num = None, time_num = None):

    for idx, ts in enumerate(tso):
        var = ts['paleoData_variableName']
        print(f'{idx} : {var}')

    if val_num is None:
        val_num = input('Select the index of the value axis: ')

        try:
            val_num = int(val_num)
        except ValueError:
            clear_output()
            print('Value index passed could not be coaxed into integer, please pass an integer \n')
            val_num,time_num = _get_tso_indices(tso)

    if time_num is None:
        time_num = input('Select the index of the time axis: ')

        try:
            time_num = int(time_num)
        except ValueError:
            clear_output()
            print('Time index passed could not be coaxed into integer, please pass an integer \n')
            val_num,time_num = _get_tso_indices(tso)
    
    if time_num == val_num:
        clear_output()
        print('Value and time indices are identical, please re-enter selections \n')
        val_num,time_num = _get_tso_indices(tso)

    return val_num, time_num

def _get_event_time(series,event_start=None,event_end=None):

    if event_start is None:

        event_start = input('When does the event begin in units of the time axis: ')

        try:
            event_start = float(event_start)
        except ValueError:
            clear_output()
            print('Event start time passed could not be coaxed into float, please pass a float or integer \n')
            event_start,event_end = _get_event_time(series)

    if event_end is None:

        event_end = input('When does the event end in units of the time axis: ')

        try:
            event_end = float(event_end)
        except ValueError:
            clear_output()
            print('Event end time passed could not be coaxed into float, please pass a float or integer \n')
            event_start,event_end = _get_event_time(series,event_start)
    
    series_slice = series.slice((event_start,event_end))

    if series_slice.time.size == 0:
        clear_output()
        print('Series sliced with these times returned an empty array, please double check entries \n')
        event_start,event_end = _get_event_time(series)

    return event_start, event_end

def _get_amp_time(series,timing=None,amp=None,id=None):

    if timing is None:

        timing = input(f'Enter {id} event timing: ')

        try:
            timing = float(timing)
        except ValueError:
            if id == 'Middle' and not timing:
                timing = None
            else:
                clear_output()
                print(f'{id} event timing passed could not be coaxed into float, please pass a float or integer \n')
                timing,amp = _get_amp_time(series,id=id)
    
    if amp is None:

        amp = input(f'Enter {id} event amplitude: ')

        try:
            amp = float(amp)
        except ValueError:
            if id == 'Middle' and not amp:
                amp = None
            else:
                clear_output()
                print(f'{id} event amplitude passed could not be coaxed into float, please pass a float or integer \n')
                timing, amp = _get_amp_time(series,timing,id=id)

    return timing, amp

def _get_event_stats(series):

    beg_timing, beg_amp = _get_amp_time(series,id='Beginning')
    mid_timing, mid_amp = _get_amp_time(series,id='Middle')
    end_timing, end_amp = _get_amp_time(series,id='End')

    if type(mid_timing) != type(mid_amp):
        clear_output()
        print('Type of mid timing and mid amp do not match, please re-enter types.')
        mid_timing, mid_amp = _get_amp_time(series,id='Middle')

    return beg_timing, beg_amp, mid_timing, mid_amp, end_timing, end_amp


def text_to_csv(load_pathname,save_pathname=None):
    '''Function to convert NOAA formatted text file to csv
    
    Parameters
    ----------
    
    load_pathname : str
        Path to text file to be loaded
        
    save_pathname : str
        Path to csv file to be saved. If none is provided, load_pathname will be used with csv extension instead of txt'''

    if save_pathname is None:
        save_pathname = load_pathname.replace('.txt','.csv')

    with open(load_pathname,'r',encoding="utf8", errors='ignore') as f:
        lines = f.readlines()
    
    f.close()

    new_lines = []
    for line in lines:
        if line[0] != '#':
            new_lines.append(line)
            
    split_lines = [re.split('\n|\t',item.strip()) for item in new_lines]

    with open(save_pathname,'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerows(split_lines)

def load_data(load_pathname):
    '''Function to load  data

    Parameters
    ----------

    load_pathname : str
        Full path to lipd file to be loaded

    Returns
    -------

    series : pyleoclim.Series
        Series object to be used for labeling and check data functions
    '''

    #Change directory shenanigans necessary due to lipd switching directories after loading
    working_dir = os.getcwd()

    file = lpd.readLipd(load_pathname)

    if file:

        tso = lpd.extractTs(file)

        os.chdir(working_dir)

        val_num,time_num = _get_tso_indices(tso)

        series = pyleo.Series(time=tso[time_num]['paleoData_values'],value=tso[val_num]['paleoData_values'],
                        time_name = tso[time_num]['paleoData_variableName'],value_name=tso[val_num]['paleoData_variableName'],
                        time_unit = tso[time_num]['paleoData_units'], value_unit=tso[val_num]['paleoData_units']).standardize().convert_time_unit('yr BP').interp(step=1)

        return series

    else:
        os.chdir(working_dir)
        return

def visualize(series,reverse=False,x_lims=None):
    '''Function to visualize data

    Parameters
    ----------

    series : pyleoclim.series
        Series object from data loading step

    reverse : bool; {True,False}
        Whether or not to flip the y axis

    x_lims : list, tuple
        Limits for the x axis (in case zooming in is required)

    '''
    series = series.convert_time_unit('yr BP')
    series = series.interp(step=1)
    fig = px.line(x=series.time, y=series.value, 
                    labels = {'x':f'{series.time_name} [{series.time_unit}]','y':f'{series.value_name} [{series.value_unit}]'}, 
                    title='Plot for labeling')

    if reverse:
        fig['layout']['yaxis']['autorange'] = "reversed"

    if x_lims:
        fig.update_layout(xaxis_range=[x_lims[0],x_lims[-1]])

    fig.show()

def label_data(series):
    '''Function to label events in timeseries data using lipd files

    Parameters
    ----------

    series : pyleo.Series
        Series object from data loading step

    Returns
    -------

    res : dict
        Dictionary of all labels created
    
    '''

    breakpoints = ['start','first','second','third']
    available_time = series.time
    start_time = min(available_time)
    end_time = max(available_time)

    res ={}

    for point in breakpoints:
        valid = False

        while not valid:
            timing = input(f'Enter {point} spline timing: ')

            try:
                timing = float(timing)
                if start_time <= timing <= end_time:
                    valid=True
                else:
                    print(f'{timing} is not within series time range of [{start_time},{end_time}].')
            except ValueError:
                if not timing:
                    timing = None
                    valid=True
                else:
                    print(f'{point} spline timing passed could not be coaxed into float, please pass a float or integer or enter nothing to skip \n')
                    
        res[point] = timing
            
    return res


def gen_fit(series,res,reverse=False,x_lims=None,v_shift = None):
    '''Function to generate and check the fit of idealized shape as its been defined

    Parameters
    ----------

    series : pyleo.Series
        Series object from data loading step

    res : dict
        Res object from data labeling step, or dict containing:
            event_start, event_end, beg_timing, beg_amp, mid_timing, mid_amp, end_timing, end_amp as keys

    reverse : bool; {True,False}
        Whether or not to flip the y axis
    
    x_lims : list, tuple
        Limits for the x axis (in case zooming in is required)

    v_shift : float
        Amount to shift event up or down to assist with fitting. Optional

    Returns
    -------

    idealized_series : pyleo.Series
        Idealized series generated using event stats. If v_shift is passed it is not included in this series

    stats : dict
        Dictionary of idealized event statistics
    
    '''

    series = series.interp(step=1)

    splines = ['first','second','third']
    stats = {}
    stats['event_start'] = res['start']

    synth_time = series.time
    pointer_index = np.where(synth_time==res['start'])[0][0]
    synth_val = np.zeros(synth_time.size) + series.value[pointer_index]

    for spline in splines:
        if res[spline]:
            begin = pointer_index
            end = np.where(synth_time==res[spline])[0][0]
            duration = float(end-begin)
            delta_amp = series.value[end]-series.value[begin]
            synth_val[int(begin):int(end)] = np.arange(duration)*(delta_amp/duration) + synth_val[begin-1]
            stats[f'{spline}_dur'] = duration
            stats[f'{spline}_amp'] = delta_amp
            pointer_index=end
        if not res[spline]:
            stats[f'{spline}_dur'] = 0
            stats[f'{spline}_amp'] = 0

    stats['event_end'] = synth_time[pointer_index]

    synth_val[int(pointer_index):] = synth_val[int(pointer_index)-1]

    if v_shift:
        synth_val += v_shift

    synth_series = pyleo.Series(synth_time,synth_val,label='Idealized event')
    interp_series = series.interp(step=1)

    fig = px.line(labels = ('{series.time_name} {series.time_units}','{series.value_name} {series.value_units}'),
                title="Idealized vs. Real Event")

    fig.add_scatter(x=interp_series.time, y=interp_series.value, name='Original Series')
    fig.add_scatter(x=synth_series.time, y = synth_series.value, name = 'Idealized Event Series')

    if reverse:
        fig['layout']['yaxis']['autorange'] = "reversed"

    if x_lims:
        fig.update_layout(xaxis_range=[x_lims[0],x_lims[-1]])

    fig.show()

    return stats

def save_data(series,stats,type_of_event,load_path,save_path,
              associated_variable_index=None,event_num=None,realization_num=None):
    '''Function to save labelled event series and idealized shape statistics

    Parameters
    ----------

    idealized_series : pyleo.Series
        Idealized version of the event in series format generated by gen_fit

    stats : dict
        Dictionary generated by gen_fit containing: beg_dur, beg_amp, mid_dur, mid_amp, end_dur, end_amp, event_start, event_end

    type_of_event : str
        Type of event recorded ['heinrich','dansgaard-oeschger','younger-dryas','8.2ka',etc.]
    
    load_path : str
        Path to original lipd file

    save_path : str
        Path to lipd file to be generated

    associated_variable_index : int
        Index of dependent variable in lipd file associated with this event entry. Will be queried for if not passed

    event_num : int
        ID to use for event. Recommended that you start with 0 and add 1 for each additional event (0,1,2,3,etc.)

    realization_num : int
        ID to use for ideal realization of event. Recommended that you start with 0 and add 1 for each additional event (0,1,2,3,etc.) 
    '''

    if event_num is None:
        event_num = input('What event num would you like to use for this event entry?: ')

    try:
        event_num = int(event_num)
    except:
        raise ValueError('Passed event num could not be coaxed into integer value, please try again.')

    if realization_num is None:
        realization_num = input('What realization num would you like to use for this realization entry?: ')

    try:
        event_num = int(event_num)
    except:
        raise ValueError('Passed realization num could not be coaxed into integer value, please try again.')

    load_file = lpd.readLipd(load_path)

    labels = np.zeros(series.time.size)
    labels[(series.time >= stats['event_start']) & (series.time <= stats['event_end'])] = 1
    label_list = list(labels)

    event_name = f'Event_{event_num}'

    if event_name in list(load_file['paleoData']['paleo0']['measurementTable']['paleo0measurement0']['columns'].keys()):

        realization_name = f'Realization_{realization_num}'

        C = {}
        C['stats'] = {}

        for key in list(stats.keys()):
            C['stats'][key] = stats[key]

        load_file['paleoData']['paleo0']['measurementTable']['paleo0measurement0']['columns'][event_name][realization_name] = C

    else:

        realization_name = f'Realization_{realization_num}'

        C = {}
        C['number'] = len(list(load_file['paleoData']['paleo0']['measurementTable']['paleo0measurement0']['columns'].keys()))+1
        C['units'] = 'NA'
        C['variableName'] = event_name
        C['variableType'] = 'inferred'
        C['event_type'] = type_of_event
        C['values'] = label_list
        C[realization_name] = {}
        C[realization_name]['stats'] = {}

        for key in list(stats.keys()):
            C[realization_name]['stats'][key] = stats[key]

        tso = lpd.extractTs(load_file) 

        if not associated_variable_index:

            for idx, ts in enumerate(tso):
                var = ts['paleoData_variableName']
                print(f'{idx} : {var}')
            
            valid = False

            while not valid:

                associated_variable_index = input('Select the index of the variable associated with this event entry: ')

                try:
                    associated_variable_index = int(associated_variable_index)
                    valid=True
                except ValueError:
                    print('Index passed could not be coaxed into integer, please pass an integer \n')   
        
        C['associated_variable_index'] = associated_variable_index
        C['associated_variable'] = tso[associated_variable_index]['paleoData_variableName']

        load_file['paleoData']['paleo0']['measurementTable']['paleo0measurement0']['columns'][event_name] = C

    lpd.validate(load_file,detailed=True)

    lpd.writeLipd(load_file,save_path)
    
def create_ideal_series(series,stats):
    '''Function to load and display a taxonomy entry
    
    Parameters
    ----------
    
    series : pyleoclim.Series
        Original series object

    stats : dict
        Dictionary containing the pieces we need to create our idealized series.
        Should be contained in 'event_metadata' portion of taxonomy entry.

    bounds : list or tuple
        List or tuple containing (start,end) integer values for series. 
        Event start and event end as contained in the stats object should be within this range
        
    Returns
    -------
    
    idealized_series : pyleo.Series
        Idealized series created from event statistics recorded in taxonomy entry'''

    series=series.interp(step=1)

    time = np.arange(series.time[0],series.time[-1],dtype=int)

    event_start = stats['event_start']

    time_pointer = int(event_start)
    value_pointer = series.value[np.where(time==time_pointer)[0][0]]

    values = np.zeros(time.size) + value_pointer

    splines = ['first','second','third']

    for spline in splines:
        amp = stats[f'{spline}_amp']
        dur = stats[f'{spline}_dur']

        if amp != 0 and dur != 0:
            f1 = time>=time_pointer
            f2 = time<time_pointer+dur
            mask = np.all([f1,f2],axis=0)
            values[mask] = np.arange(dur)*(amp/dur) + value_pointer
            time_pointer+=dur
            value_pointer+=amp
        else:
            continue

    values[time>=time_pointer] = values[time==(time_pointer-1)]

    idealized_series = pyleo.Series(time=time,value=values)
    
    return idealized_series

    