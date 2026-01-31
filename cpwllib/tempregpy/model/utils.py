import pandas as pd
from .constants import *
from ..logging_config import *
from pandas.tseries.holiday import USFederalHolidayCalendar

def is_holiday(date): 
    """
    Check if the given date is a holiday.
    """
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=date, end=date)
    return not holidays.empty

def process_dates_to_groups(dates):
    date_groups = []

    global_counter = 0

    sunday_counter = 0
    yesterday = pd.to_datetime("1800-01-01") 

    for _date in dates:
        
        if _date.weekday() == 6:  # Sunday
            global_counter += 1
            date_groups.append(global_counter)
            sunday_counter = global_counter
            global_counter += 1
            
        elif is_holiday(_date):
            if (_date.month != yesterday.month or _date.year != yesterday.year):
                sunday_counter += 1

            date_groups.append(sunday_counter)
            global_counter += 1

        # elif _date.weekday() < 5:  # Monday to Friday
        #     if _date in [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-02')]:  # Specific dates
        #         date_groups.append(sunday_counter)
        #     else:
        #         date_groups.append(global_counter)

        elif _date.weekday() < 5:  # Monday to Friday
            # 1) If we crossed a month or year boundary since yesterday → new group
            if (_date.month != yesterday.month) or (_date.year != yesterday.year):
                global_counter += 1
                date_groups.append(global_counter)

            # 2) Your existing “special‐dates” hack lives next
            elif _date in (pd.Timestamp('2023-01-01'),
                           pd.Timestamp('2023-01-02')):
                date_groups.append(sunday_counter)

            # 3) Otherwise stay in the current group
            else:
                date_groups.append(global_counter)

            yesterday = _date


        elif _date.weekday() == 5:  # Saturday
            global_counter += 1
            date_groups.append(global_counter)
        


    return date_groups
