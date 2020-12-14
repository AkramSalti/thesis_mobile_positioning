import pandas as pd
import numpy as np
import heapq
from scipy.stats import norm
from itertools import combinations, product
import geopy.distance
from collections import Counter
import concurrent.futures
from tqdm.notebook import tqdm

def mydist(p1,p2):
    distance=geopy.distance.distance(p1,p2).meters
    return distance

def CoordToGridcell(Lat, Long, Resolution: int = 100):
    """Assigns the corresponding Grid Cell to the Position (lat, long).
        In harmony with the resolution/length of the Grid

    Args:
        Lat (int64): The Latitude coordinate of the Position multiplied by 10**6.
        Long (int64): The Longitude coordinate of the Position multiplied by 10**6.
        Resolution (int, optional): The length of the sides of the Grid Cells. Defaults to 100.

    Returns:
        tuple (int64,int64): The Latitude and Longitude coordinates of the corresponding Grid Cell.
    """
    lat_cell = np.int64(Lat - (Lat % Resolution) + Resolution / 2) 
    long_cell = np.int64(Long - (Long % Resolution) + Resolution / 2) 
    return lat_cell, long_cell


def getRSRPfrom_Measurement(Measurement, measurement_type: str = 'outdoor', **kwargs):
    """Wrapper for RSRP extraction.

    Args:
        Measurement (dictionary): A dictionary containing the measured Cell IDs and Signal Strengths.
        measurement_type (str, optional): The format of the measurements. Defaults to 'outdoor'.

    Returns:
        dictionary: The result of the extraction
    """
    if measurement_type == 'outdoor':
        return getRSRPfrom_outdoorMeasurement(Measurement)
    elif measurement_type == 'indoor':
        return getRSRPfrom_indoorMeasurement(Measurement, **kwargs)
    else:
        return dict()


def getRSRPfrom_outdoorMeasurement(Measurement, **kwargs):
    """Extracts the Cell ID, Signal Strength pairs from a dictionary.

    Args:
        Measurement (dictionary): A dictionary containing the measured Cell IDs and Signal Strengths.
        The relevant keys in the outdoor measurement are the following:
        CID-S 	RSRP-S 	CID-N1 	RSRP-N1 	CID-N2 	RSRP-N2 	CID-N3 	RSRP-N3 	CID-N4 	RSRP-N4 	CID-N5 	RSRP-N5 	CID-N6 	RSRP-N6
        CID-[S|1-6] and RSRP-[S|1-6] in regex

    Returns:
        dictionary: The Cell IDs are the keys and the Signal Strengths are the values.
    """
    extracted_RSRP = {}
    for key in Measurement.keys():
        if 'CID' in key:
            CID = Measurement[key]
            if pd.isnull(CID):
                continue
            key_rsrp = 'RSRP-' + key.split('-')[1]
            RSRP = Measurement[key_rsrp]
            extracted_RSRP[CID] = RSRP
    return extracted_RSRP

def getRSRPfrom_indoorMeasurement(Measurement, deviation_threshold: int = 28, **kwargs):
    """Extracts the PID, Signal Strength pairs from a dictionary.

    Args:
        Measurement (dictionary): A dictionary containing the measured PIDs and Signal Strengths.
        deviation_threshold (int): Measurements with a deviation higher than this threshold are discarded. (To much noise)
        The relevant key format in the indoor measurement are the following:
        PID\d, RSRP\d, RSRP-D\d

    Returns:
        dictionary: The Cell IDs are the keys and the Signal Strengths are the values.
    """
    extracted_RSRP = {}
    for key in Measurement.keys():
        if 'PID' in key:
            CID = Measurement[key]
            if pd.isnull(CID):
                continue
            key_rsrp = 'RSRP' + key[-1]
            key_deviation = 'RSRP-D' + key[-1]
            RSRP = Measurement[key_rsrp]
            if key_deviation in Measurement and (Measurement[key_deviation] <= deviation_threshold):
                extracted_RSRP[CID] = RSRP
            elif key_deviation not in Measurement:
                extracted_RSRP[CID] = RSRP
    return extracted_RSRP


def getPointsfrom_Measurement(Measurement, measurement_type: str = 'outdoor', **kwargs):
    """Wrapper for Points calculation.

    Args:
        Measurement (dictionary): A dictionary containing the measured Cell IDs and Signal Strengths.
        measurement_type (str, optional): The format of the measurements. Defaults to 'outdoor'.

    Returns:
        Counter: The result of the point calculation
    """
    if measurement_type == 'outdoor':
        return getPointsfrom_outdoorMeasurement(Measurement, **kwargs)
    elif measurement_type == 'indoor':
        return Counter(getRSRPfrom_indoorMeasurement(Measurement, **kwargs))
    else:
        return Counter()



def getPointsfrom_outdoorMeasurement(Measurement, pointsToGive=[3,2,1], **kwargs):
    """Based on the Signal Strength ranking of the measurement, the Cell IDs receive points.
        In the default case, 3 for the Serving, 2 for the 1st Neighbor and 1 for the 2nd.
        It is used to create a Cell ID ranking in a Grid Cell, which is used later by Cell Selection

    Args:
        Measurement (dictionary): A dictionary containing the measured Cell IDs and Signal Strengths.
        pointsToGive (list, optional): The list of the received points. Defaults to [3,2,1].

    Returns:
        Counter: A Counter object, basicly a dictionary, but it can add up
    """
    points = Counter()
    for idx, key in enumerate(['S', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6']):
        CID = Measurement['CID-'+key]
        if not pd.isnull(CID) and idx < len(pointsToGive):
            points[CID] = pointsToGive[idx]
    return points


def getPointsfrom_indoorMeasurement(Measurement, pointsToGive=[3,2,1], **kwargs):
    """Based on the Signal Strength ranking of the measurement, the Cell IDs receive points.
        In the default case, 3 for the Serving, 2 for the 1st Neighbor and 1 for the 2nd.
        It is used to create a Cell ID ranking in a Grid Cell, which is used later by Cell Selection

    Args:
        Measurement (dictionary): A dictionary containing the measured Cell IDs and Signal Strengths.
        pointsToGive (list, optional): The list of the received points. Defaults to [3,2,1].

    Returns:
        Counter: A Counter object, basicly a dictionary, but it can add up
    """
    points = Counter()
    for idx in range(8):
        CID = Measurement['PID'+str(idx+1)]
        if not pd.isnull(CID) and idx < len(pointsToGive):
            points[CID] = pointsToGive[idx]
    return points


def calculateEstimatedPosition(dfr, useCellsforEstimation: int = 10):
    """Calculating the estimated position using the Similarity Scores of the top N (useCellsforEstimation) Grid Cells.
        The caculation is performed with the matrixes of the Positions and Similarity Scores.

    Args:
        dfr (DataFrame): Contains the position of the measurement and the Grid Cells with the 50 highest Similarity Score
                        Score_0, Cell_0 and Error_0 describes the best matching cell
                        Score_1, Cell_0 and Error_1 ...
        useCellsforEstimation (int, optional): The number of Grid Cells used at the position estimation. Defaults to 10.

    Returns:
        Lats, Longs [numpy arrays]: The estimated Lat, Long coordinates in numpy arrays
    """
    S = np.nan_to_num(dfr.filter(regex='Score_\d').to_numpy().astype(float))[:,:useCellsforEstimation]
    W = S/S.sum(axis=1, keepdims=True)
    C = np.array(dfr.filter(regex='Cell_\d').applymap(lambda x: (0,0) if x is np.nan else x), dtype=[('Lat', 'i8'), ('Long', 'i8')])[:,:useCellsforEstimation]
    #.fillna(value='(0,0)').applymap(literal_eval)
    Lats = np.multiply(W,C['Lat']).sum(axis=1, keepdims=False) / 10**6
    Longs = np.multiply(W,C['Long']).sum(axis=1, keepdims=False) / 10**6
    return Lats, Longs


def calculateEstimatedPositions(dfr, to_weight):
    """Calculates the estimated positions using different number of top Grid Cells given in to_weight.
        Calculates the 2D Errors in meters for the estimated positions.
        The 'w' in the column names will indicate estimated values.

    Args:
        dfr (DataFrame): Contains the position of the measurement and the Grid Cells with the 50 highest Similarity Score
                        Score_0, Cell_0 and Error_0 describes the best matching cell
                        Score_1, Cell_0 and Error_1 ...
        to_weight (list): The list of the number of Grid Cells used at the estimated position calculation.

    Returns:
        dfr [DataFrame]: The complete DataFrame with the estimated positions
    """
    for i in to_weight:
        Lat, Long = calculateEstimatedPosition(dfr,i)
        Error_W = []
        for index, row in dfr.iterrows():
            if np.isnan(Lat[index]):
                Error_W.append(np.nan)
            else:
                Error_W.append(geopy.distance.distance((row['Lat']/10**6, row['Long']/10**6), (Lat[index], Long[index])).m)
        dfr['Error_w'+str(i)] = Error_W
        dfr['Lat_w'+str(i)] = Lat
        dfr['Long_w'+str(i)] = Long
    return dfr
