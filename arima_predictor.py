import os
import warnings
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

def concat_csv(data_path:str, keyword:str, use_cols:list[str]) -> pd.DataFrame:
    """
    Concatenates CSV files in a directory containing a specific keyword, using specified columns.
    """

    # Filter files containing a keyword
    files = [file for file in os.listdir(data_path) if (keyword in file) and ('d3' not in file)]
    
    # Read files and put them in a list
    df_list = [pd.read_csv(f'{data_path}/{file}', usecols=use_cols) for file in files]

    # Concat the list to have only one dataframe
    df = pd.concat(df_list)

    return df


def complete_time_series(time_series:pd.DataFrame, obj_var:str='CLOSE') -> pd.Series:
    """
    Completes a time series by adding missing hourly data points between the minimum and maximum dates.

    Args:
        time_series: The input time series DataFrame with a DatetimeIndex.
        obj_var: The name of the column containing the values of interest (default: 'CLOSE').

    Returns:
        A Pandas Series with the completed time series, including any missing hourly data points.
    """
    # Extract min and max date
    min_date = time_series.index.min()
    max_date = time_series.index.max()

    # Generate complete time series 
    time_index = pd.date_range(min_date, max_date, freq='H', inclusive='right')
    time_index = pd.DataFrame(index=time_index)

    # merge to add all records to series
    time_series = pd.merge(time_series, time_index, how='outer', left_index=True, right_index=True)
    
    return time_series[obj_var]


def tune_order_arima(q:int, time_series:pd.Series):
    """
    Calculates the Mean Absolute Error (MAE) for experiments with order parameter.

    Args:
        q: The total number of data points to use (determines the train/test split).
        time_series: The time series data as a Pandas Series with a DatetimeIndex and hourly frequency.

    Returns:
        the best combination of order parameter
    """
    # Split into train and test sets
    test_size = int(q * 0.1)
    train, test = time_series[-q:-test_size], time_series[-test_size:]
    
    best_mae = float('inf')
    best_order = None
    for p in range(3):
        for d in range(2):
            for q in range(3):
                model = ARIMA(train, order=(p, d, q))
                model_fit = model.fit()
                forecast_val = model_fit.forecast(steps=len(test))
                mae_val = mean_absolute_error(test, forecast_val)
                if mae_val < best_mae:
                    best_mae = mae_val
                    best_order = (p, d, q)

    return best_order


def train_final_arima_model(q:int, fh:int, order:tuple[int]=(1, 0, 1)) -> pd.DataFrame:
    """
    Trains an ARIMA model on the last 'q' data points of a time series, 
    generates a forecast for 'fh' steps, and returns the forecasted values, 
    confidence intervals, and standard errors in a DataFrame.

    Args:
        q: Number of last data points to use for training.
        fh: Forecast horizon (number of steps to forecast).

    Returns:
        DataFrame containing forecasted values, confidence intervals, and standard errors.
    """
    # Retrain with whole data
    model = ARIMA(time_series[-q:], order=order)  # Replace with your order
    model_fit = model.fit()

    # Get the forecast for 10 steps ahead
    forecast_object = model_fit.get_forecast(steps=fh)

    # Get the forecasted values, intervals and s
    forecast_values = forecast_object.predicted_mean
    confidence_intervals = forecast_object.conf_int()
    standard_error = forecast_object.se_mean

    # Concat generated info into a dataframe
    df_list = [forecast_values, confidence_intervals] #, standard_error] 
    final_pred = pd.concat(df_list, axis=1)
    
    # Rename columns to match format
    rename_dict = {
        'predicted_mean': 'Prediction', 
        'lower CLOSE': 'LB', 
        'upper CLOSE': 'UB', 
    }
    final_pred.rename(columns=rename_dict, inplace=True)

    return final_pred


def preprocess_data_to_time_series(df:pd.DataFrame, 
                                   col_date:str='DATE_TIME', 
                                   col_obj:str='CLOSE') -> pd.Series:
    
    # Convert dataframe to time series
    series = pd.to_datetime(df[col_date])
    df.set_index(series, inplace=True)
    time_series = df[col_obj]

    # fill data from weekends with last known from friday
    time_series = complete_time_series(time_series)
    time_series = time_series.ffill()
    return time_series


def generare_results_format(final_results:pd.DataFrame) -> pd.DataFrame: 

    ## Shift results to get 6 hour predictions in a row
    results = []
    for i in range(1, 7):
        aux_df = final_results.shift(-(i - 1)).copy()
        aux_df.columns = [f'{c}_{i}' for c in aux_df]
        results.append(aux_df)

    ## Consolidate results
    results = pd.concat(results, axis=1)
    results = results.dropna()
    
    return results


if __name__ == '__main__':
    
    # Read data from path
    data_path = 'examples/data/phase_1/'
    use_cols = ['DATE_TIME', 'CLOSE']
    df = concat_csv(data_path=data_path, 
                    keyword='normalized', # 'base' | 'normalized'
                    use_cols=use_cols)
    
    time_series = preprocess_data_to_time_series(df)

    # make experiments
    data_quantity = [1575, 3150, 6300, 12600, 25200]
    orders = [(1, 1, 2), (2, 1, 2), (0, 1, 0), (2, 1, 1), (0, 1, 0)]
    max_steps_test = 6300
    for i, q in enumerate(data_quantity):
        # mae = get_mae_train_arima_model(q, time_series)
        if not orders:
            best_order = tune_order_arima(q=q, time_series=time_series)
        else:
            best_order = orders[i]
        print('*' * 80, f'\ndata_quantity: {q:,.0f} with order: {best_order}')
        final_results = train_final_arima_model(q=q, fh=max_steps_test, order=best_order)
        results = generare_results_format(final_results)


        # Save results
        save_path = 'examples/results/phase_1'
        file_name = f'phase_1_arima_{q}_1h_prediction.csv'
        results.to_csv(f'{save_path}/{file_name}', index=False)