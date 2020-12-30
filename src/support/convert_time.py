import datetime

def convert_date_string_to_unix_seconds(date_and_time):
    """
    Convert a date & time to a unix timestamp in seconds.

    Input:
        date_and_time (str): date and time (Toronto time zone) in the
            'YYYY_MM_DD_hh_mm_ss_microsec' format
    Return:
        float: equivalent unix timestamp, in seconds
    """
    # Extract microseconds part and convert it to seconds
    microseconds_str = date_and_time.split('_')[-1]
    seconds_remainder = float('0.' + microseconds_str)
    
    # Extract date without microseconds and convert to unix timestamp
    # The added 'GMT-0400' indicates that the provided date is in the 
    # Toronto (eastern) timezone during daylight saving
    date_to_sec_str = date_and_time.replace('_' + microseconds_str, '')
    seconds_to_sec = \
        datetime.datetime.strptime(date_to_sec_str + \
        ' GMT-0400', "%Y_%m_%d_%H_%M_%S GMT%z").timestamp()

    # Add the microseconds remainder
    return seconds_to_sec + seconds_remainder