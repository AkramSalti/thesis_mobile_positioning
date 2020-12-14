import csv
import pandas as pd

from pandas import DataFrame,options
import numpy as np
#options.mode.chained_assignment = None


def getDataFromCSV(filePath):
    tsv_file = open(filePath)
    read_tsv = csv.DictReader(tsv_file, dialect='excel-tab')
    measurements = []
    for row in read_tsv:
        measurements.append(row)
    return measurements

def getDataFrameBasedOnParameters(List, Parameters):
    measurementsList = []
    for i in range(len(List)):
     if (len(Parameters)==1):
           measurementsList.append([List[i].get(Parameters[0])])
     elif (len(Parameters) == 2):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1])])
     elif (len(Parameters) == 3):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2])])
     elif (len(Parameters) == 4):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]),List[i].get(Parameters[3])])
     elif (len(Parameters) == 5):

           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4])])
     elif (len(Parameters) == 6):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5])])
     elif (len(Parameters) == 7):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6])])
     elif (len(Parameters) == 8):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]),List[i].get(Parameters[7])])
     elif (len(Parameters) == 9):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]),List[i].get(Parameters[7]),
                                    List[i].get(Parameters[8])])
     elif (len(Parameters) == 10):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]),List[i].get(Parameters[7]),
                                    List[i].get(Parameters[8]),List[i].get(Parameters[9])])
     elif (len(Parameters) == 11):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]),List[i].get(Parameters[7]),
                                    List[i].get(Parameters[8]), List[i].get(Parameters[9]),
                                    List[i].get(Parameters[10])])
     elif (len(Parameters) == 12):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]), List[i].get(Parameters[7]),
                                    List[i].get(Parameters[8]), List[i].get(Parameters[9]),
                                    List[i].get(Parameters[10]),List[i].get(Parameters[11])])
     elif (len(Parameters) == 13):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]), List[i].get(Parameters[7]),
                                    List[i].get(Parameters[8]), List[i].get(Parameters[9]),
                                    List[i].get(Parameters[10]),List[i].get(Parameters[11]),
                                    List[i].get(Parameters[12])])
     elif (len(Parameters) == 14):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]), List[i].get(Parameters[7]),
                                    List[i].get(Parameters[8]), List[i].get(Parameters[9]),
                                    List[i].get(Parameters[10]),List[i].get(Parameters[11]),
                                    List[i].get(Parameters[12]),List[i].get(Parameters[13])])
     elif (len(Parameters) == 15):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]), List[i].get(Parameters[7]),
                                    List[i].get(Parameters[8]), List[i].get(Parameters[9]),
                                    List[i].get(Parameters[10]), List[i].get(Parameters[11]),
                                    List[i].get(Parameters[12]), List[i].get(Parameters[13]),
                                    List[i].get(Parameters[14])])
     elif (len(Parameters) == 16):
           measurementsList.append([List[i].get(Parameters[0]), List[i].get(Parameters[1]),
                                    List[i].get(Parameters[2]), List[i].get(Parameters[3]),
                                    List[i].get(Parameters[4]), List[i].get(Parameters[5]),
                                    List[i].get(Parameters[6]), List[i].get(Parameters[7]),
                                    List[i].get(Parameters[8]), List[i].get(Parameters[9]),
                                    List[i].get(Parameters[10]), List[i].get(Parameters[11]),
                                    List[i].get(Parameters[12]), List[i].get(Parameters[13]),
                                    List[i].get(Parameters[14]),List[i].get(Parameters[15])])

    measurementsDataFrame = DataFrame(measurementsList, columns=Parameters,dtype=float)
    return measurementsDataFrame


def convertDataFrameForPositioning(df):
    df=DataFrame(df)
    df['Lat'] = pd.to_numeric(df['Lat'], errors='coerce')
    df['Lat'].astype('float')
    df['Long'] = pd.to_numeric(df['Long'], errors='coerce')
    df['Long'].astype('float')
    df['CID-S'] = pd.to_numeric(df['CID-S'], errors='coerce')
    df['CID-S'].astype('float')
    df['CID-N1'] = pd.to_numeric(df['CID-N1'], errors='coerce')
    df['CID-N1'].astype('float')
    df['CID-N2'] = pd.to_numeric(df['CID-N2'], errors='coerce')
    df['CID-N2'].astype('float')
    df['CID-N3'] = pd.to_numeric(df['CID-N3'], errors='coerce')
    df['CID-N3'].astype('float')
    df['CID-N4'] = pd.to_numeric(df['CID-N4'], errors='coerce')
    df['CID-N4'].astype('float')
    df['CID-N5'] = pd.to_numeric(df['CID-N5'], errors='coerce')
    df['CID-N5'].astype('float')
    df['CID-N6'] = pd.to_numeric(df['CID-N6'], errors='coerce')
    df['CID-N6'].astype('float')
    df['RSRP-S'] = pd.to_numeric(df['RSRP-S'], errors='coerce')
    df['RSRP-S'].astype('float')
    df['RSRP-N1'] = pd.to_numeric(df['RSRP-N1'], errors='coerce')
    df['RSRP-N1'].astype('float')
    df['RSRP-N2'] = pd.to_numeric(df['RSRP-N2'], errors='coerce')
    df['RSRP-N2'].astype('float')
    df['RSRP-N3'] = pd.to_numeric(df['RSRP-N3'], errors='coerce')
    df['RSRP-N3'].astype('float')
    df['RSRP-N4'] = pd.to_numeric(df['RSRP-N4'], errors='coerce')
    df['RSRP-N4'].astype('float')
    df['RSRP-N5'] = pd.to_numeric(df['RSRP-N5'], errors='coerce')
    df['RSRP-N5'].astype('float')
    df['RSRP-N6'] = pd.to_numeric(df['RSRP-N6'], errors='coerce')
    df['RSRP-N6'].astype('float')
    return df

def reindexDataFrame(df):
    dfIndexed=DataFrame()
    dfIndexed['Lat'] = df['Lat'].tolist()
    dfIndexed['Long'] = df['Long'].tolist()
    dfIndexed['CID-S'] = df['CID-S'].tolist()
    dfIndexed['CID-N1'] = df['CID-N1'].tolist()
    dfIndexed['CID-N2'] = df['CID-N2'].tolist()
    dfIndexed['CID-N3'] = df['CID-N3'].tolist()
    dfIndexed['CID-N4'] = df['CID-N4'].tolist()
    dfIndexed['CID-N5'] = df['CID-N5'].tolist()
    dfIndexed['CID-N6'] = df['CID-N6'].tolist()
    dfIndexed['RSRP-S'] = df['RSRP-S'].tolist()
    dfIndexed['RSRP-N1'] = df['RSRP-N1'].tolist()
    dfIndexed['RSRP-N2'] = df['RSRP-N2'].tolist()
    dfIndexed['RSRP-N3'] = df['RSRP-N3'].tolist()
    dfIndexed['RSRP-N4'] = df['RSRP-N4'].tolist()
    dfIndexed['RSRP-N5'] = df['RSRP-N5'].tolist()
    dfIndexed['RSRP-N6'] = df['RSRP-N6'].tolist()
    return dfIndexed

def getLatList(filePath):
    measurements=getDataFromCSV(filePath)
    Lat = []
    for i in range(len(measurements) - 1):
        Lat.append(float(measurements[i].get('Lat')))
    return Lat
