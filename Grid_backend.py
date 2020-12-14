from Unified_functions import *


class GridCell:
    """The class of the Grid Cell

    Shared variables:
        nextCellID [int]: The id of the next initialized cell, increased when used.
        pointsToGive [list of ints]: The points given to the "top" CIDs of a measurement (Serving, Neighbor-1, Neighbor-2 in this case)

    Own variables of the Cells:
        id [int]: A basic int id.
        lat, long [int]: The Latitude and Longitude coordinates of the cell multiplied by 10**6 and stored as int64.
                            The 'CoordToGridcell' function calculates the centre of the cell using the coordinates and the Grid Cell length (resolution)
        measurements [dictionary]: Dictionary for the lists of the measured Signal Strengths
                                    The CIDs are the keys and the values are the lists of the measured Signal Strengths belonging to that CID.
                                    for example: {11:[-88,-86,-93],22:[-90,-91,-96],33:[-100]}
        pdf [dictionary]: Dictionary for the Gaussian distribution parameters (expected value and variance)
                            The CIDs are the keys and the values are the parameters of the Gaussian distribution fitted to the measured Signal Strengths belonging to that CID.
                            for example: {11:(-88.5,3),22:(-92,3),33:(-100,1)}
        measurement_num [int]: The number of measurements in the Grid Cell.
        points [Counter]: Storing the given points to the CIDs, the Counter object is basicly a dictionary but can be added together
    """
    nextCellID = 1
    pointsToGive = [3, 2, 1]
    measurement_type = None

    def __init__(self, Measurement, GridCellCenter: tuple = None, pointsToGive: list = [3, 2, 1], Resolution: int = 100,
                 useDifferential: bool = False, measurement_type: str = 'outdoor'):
        """Initializing the Grid Cell based on a measurement.

        Args:
            Measurement (dictionary): Contains the position of the measurement and the Cell IDs and Signal Strengths
            pointsToGive (list, optional): The 'impact' points of the top CIDs. Defaults to [3,2,1].
        """
        # * Setting the id and calculating the centre of the cell
        self.id = GridCell.nextCellID
        GridCell.nextCellID += 1
        GridCell.measurement_type = measurement_type
        if not GridCellCenter:
            self.lat, self.long = CoordToGridcell(Measurement['Lat'], Measurement['Long'], Resolution=Resolution)
        else:
            self.lat, self.long = GridCellCenter

        # * The main dictionaries
        self.measurements = {}
        self.pdf = {}
        self.measurements_num = 1
        self.using_Differential = useDifferential
        self.diff_measurements = {}
        self.diff_pdf = {}

        # * Initializing the 'impact' points based on the first measurement
        GridCell.pointsToGive = pointsToGive
        self.points = getPointsfrom_Measurement(Measurement, measurement_type=GridCell.measurement_type,
                                                **{'pointsToGive': GridCell.pointsToGive})

        # * Initializing the first Signal Strength lists
        extracted_RSRP = getRSRPfrom_Measurement(Measurement, measurement_type=GridCell.measurement_type)
        for CID, RSRP in extracted_RSRP.items():
            self.measurements[CID] = [RSRP]

    def update(self, Measurement):
        """Updating the Grid Cell with another measurement

        Args:
            Measurement (dictionary): Contains the position of the measurement and the Cell IDs and Signal Strengths
        """
        # *Increasing the number of measurements in the cell
        self.measurements_num += 1

        # * Updating the 'impact' points with the new measurement
        self.points += getPointsfrom_Measurement(Measurement, measurement_type=GridCell.measurement_type,
                                                 **{'pointsToGive': GridCell.pointsToGive})

        # * Updating or initializing the Signal Strength lists
        extracted_RSRP = getRSRPfrom_Measurement(Measurement, measurement_type=GridCell.measurement_type)
        for CID, RSRP in extracted_RSRP.items():
            if CID not in self.measurements:
                self.measurements[CID] = [RSRP]
            else:
                self.measurements[CID].append(RSRP)

    def fitGaussianDistribution(self, min_variance: float = 1.0):
        """Calculating the Gaussian function parameters of the Signal Strength's distributions
            The scipy.stats package is used
            The Gaussian distribution is described with 2 parameters: Expected value and Variance

        Args:
            min_variance (float, optional): The minimum variance to use. Low number of measurements can result in 0 or very low variance. Defaults to 1.0.
        """
        # * For each occured CID a Gaussian function is fitted to the measured Signal Strengths
        for CID in self.measurements:
            E, V = norm.fit(self.measurements[CID])
            # * If the variance would be lower than the minimum accepted value, than it is exchanged.
            if V < min_variance:
                V = min_variance
            self.pdf[CID] = (E, V)

    def calculateSimilarityScore(self, Measurement, unknownCell_punishment: float = 0.01,
                                 knownCell_bonus: float = 500.0, lowCIDCount_punishment: bool = False,
                                 shiftRSRP: bool = False, useAbsolute: bool = True, useDifferential: bool = False,
                                 differential_topN: int = 2):
        """Calculating the Similarity Score between a measurement and the Grid Cell.
            The main laws:
                1. For each CID in the measurement that DID NOT occured in the Grid Cell before, a punishment is applied
                2. For each CID in the measurement that occured in the Grid Cell before, the measured Signal Strength is **behelyettesit** in the CID's distribution function and a reward is applied
                3. For each CID that occured in the Grid Cell before but not in the measurement a punishment applied based on the frequency of that CID

        Args:
            Measurement (dictionary): Contains the position of the measurement and the Cell IDs and Signal Strengths
            unknownCell_punishment (float, optional): The punishment of an unknown (not occured in the cell before) CID in the measurement. Defaults to 0.01.
            knownCell_bonus (float, optional): The bonus of a known CID (occured in the cell before). Defaults to 500.0.
            lowCIDCount_punishment (bool, optional): The 3rd law is optional. If the offline measurements contain more CIDs than the online measurements than it's better not to use it. Defaults to True.
            shiftRSRP (bool, optional): A test feature that shifts the Signal Strength values with the difference of the Serving Cell's Signal Strength and it's expected value. Defaults to False.
            useAbsolute (bool optional): Whether to use the absolute RSRP values at Similarity Score Calculation. Defaults to True.
            useDifferential (bool optional): Whether to use the absolute RSRP values at Similarity Score Calculation. Defaults to False.
            differential_topN (int): Sets how much of the Signal Strengths are used for differential value calculation. Defaults to 2.
        """
        # * The base Similarity Score is 1
        P = 1.0
        unknownCells_number = 0
        knownCells_number = 0
        received_CIDs = []
        extracted_RSRP = getRSRPfrom_Measurement(Measurement, measurement_type=GridCell.measurement_type)

        # * Applying the laws
        if useAbsolute:
            for CID, RSRP in extracted_RSRP.items():
                # * 1st law (unknown CIDs)
                if CID not in self.pdf:
                    P *= 1.0
                    unknownCells_number += 1
                # * 2nd law (known CIDs)
                else:
                    received_CIDs.append(CID)
                    E, V = self.pdf[CID]
                    P *= norm.pdf(RSRP, E, V) * knownCell_bonus
                    knownCells_number += 1

        return P, unknownCells_number, knownCells_number


class Grid:
    """The class of the Grid, this is the so called Fingerprint database

    Variables:
        grid [dictionary]: The dictionary containing the Grid Cells. The keys are the tuples of the coordinates of the centers of the Grid Cells (*10**6, int64)
        cellSelection [dictionary]: Dictionary of Grid Cell sets (not actual objects just coordinates) For each CID the Grid Cells are strored where the CID is in the top X in the Grid Cell's 'impact' list.
        measurement_type [str]: Defines the format of the measurements
    """

    def __init__(self, df, GridCellCenters: list = [], pointsToGive=[3, 2, 1], Resolution: int = 100,
                 useDifferential: bool = False, measurement_type: str = 'outdoor'):
        """Initializing the Grid/Fingerprint database with the measurements in the df DataFrame and setting the Grid Cell length (Resolution) and the impact points (pointsToGive)

        Args:
            df (DataFrame): The pandas DataFrame containing the measurements
            pointsToGive (list, optional): The points given to the "top" CIDs of a measurement (Serving, Neighbor-1, Neighbor-2 in the default case). Defaults to [3,2,1].
            Resolution (int, optional): The length of the sides of the Grid Cells. Defaults to 100.
        """
        # * Initializing the grid
        self.grid = {}
        self.measurement_type = measurement_type

        # * Processing the measurements one by one, the tqdm provides a nice progress bar
        df = df.reset_index(drop=True)
        for idx, Measurement in tqdm(df.iterrows(), total=df.shape[0]):
            # * If no predefined groups are available
            # * The measurement's position and the Grid's resolution determines the Grid Cell
            if idx >= len(GridCellCenters):
                Lat, Long = CoordToGridcell(Measurement['Lat'], Measurement['Long'], Resolution=Resolution)
            else:
                Lat, Long = GridCellCenters[idx]
            # * If the Grid Cell does not exist, a new is created from the measurement
            if (Lat, Long) not in self.grid:
                self.grid[(Lat, Long)] = GridCell(Measurement, GridCellCenter=(Lat, Long), pointsToGive=pointsToGive,
                                                  Resolution=Resolution, useDifferential=useDifferential,
                                                  measurement_type=measurement_type)
            # * If the Grid Cell exist, we update it with the measurement
            else:
                self.grid[(Lat, Long)].update(Measurement)

    # A cellSelection tartalmazza, hogy mely CID-ek mely cellákban "jelentősek" (sokszor Serving vagy Neighbour 1)
    # A most_common paraméter szabályozza, hogy cellánként hány jelentős CID-et jegyzünk fel
    def calculateCellSelection(self, most_common: int = 2):
        """The cellSelection is a dictionary of Grid Cell sets (not actual objects just coordinates).
            For each CID the Grid Cells are strored where the CID is in the top X (most_common) in the Grid Cell's 'impact' list.

        Args:
            most_common (int, optional): The number of most important CIDs to select in every Grid Cell. Defaults to 2.
                                        Higher most_common means longer lists and higher runtime.
        """
        self.cellSelection = {}
        # * Iterating over the Grid Cells and checking the top CIDs
        for cell in self.grid:
            for CID, _ in self.grid[cell].points.most_common(most_common):
                # * Then the Grid Cells are added to the CID's Grid Cell set.
                if CID not in self.cellSelection:
                    self.cellSelection[CID] = set([cell])
                else:
                    self.cellSelection[CID].add(cell)

    def fitGaussianDistributions(self, min_variance: float = 1.0):
        """Fitting the Gaussian distribution to the Grid Cells' CIDs' Signal Strength lists.
            Only calls the Grid Cell's fitGaussianDistribution method

        Args:
            min_variance (float, optional): The minimum variance to use. Low number of measurements can result in 0 or very low variance. Defaults to 1.0.
        """
        for cell in self.grid:
            self.grid[cell].fitGaussianDistribution(min_variance=min_variance)

    def localization_DataFrame(self, df, cell_database=pd.DataFrame([]), RTT_precision_plus: float = 100.0,
                               RTT_precision_minus: float = 100.0, unknownCell_punishment: float = 0.01,
                               knownCell_bonus: float = 500.0, most_common: int = 2,
                               lowCIDCount_punishment: bool = False, shiftRSRP: bool = False, max_workers: int = 1,
                               useAbsolute: bool = True, useDifferential: bool = False, differential_topN: int = 2,
                               **kwargs):
        """Positioning every measurement in a DataFrame. We can set every parameter connected to the positioning.
            This fuction just calles the localization_Measurement function for every measurement.

        Args:
            df (DataFrame): The DataFrame containing the measurements and positions
            unknownCell_punishment (float, optional): The punishment of an unknown (not occured in the cell before) CID in the measurement. Defaults to 0.01.
            knownCell_bonus (float, optional): The bonus of a known CID (occured in the cell before). Defaults to 500.0.
            most_common (int, optional): The strongest CIDs in the measurement to use at the selection of the postential cells. Defaults to 2.
            lowCIDCount_punishment (bool, optional): The usage 3rd law is optional. If the offline measurements contain more CIDs than the online measurements than it's better not to use it.. Defaults to True.
            shiftRSRP (bool, optional): A test feature that shifts the Signal Strength values with the difference of the Serving Cell's Signal Strength and it's expected value. Defaults to False.
            max_workers (int, optional): The positioning of the measurements can be parallelized. If you set the max_workers parameter >1, the program will use more cores and finishes faster. To see the gain, test it with max_workers from 1 to 4. Defaults to 1.

        Returns:
            [DataFrame]: The result is a DataFrame containing the id and position of the measurement and the top 50 Grid Cells based on the Similarity Score and the corresponding Similarity Scores and 2D Errors
        """
        results = []
        # * No parallelization
        for index, row in tqdm(df.iterrows(), total=len(df)):
            results.append(self.localization_Measurement(index, row, cell_database=cell_database,
                                                         RTT_precision_plus=RTT_precision_plus,
                                                         RTT_precision_minus=RTT_precision_minus,
                                                         unknownCell_punishment=unknownCell_punishment,
                                                         knownCell_bonus=knownCell_bonus, most_common=most_common,
                                                         lowCIDCount_punishment=lowCIDCount_punishment,
                                                         shiftRSRP=shiftRSRP, useAbsolute=useAbsolute,
                                                         useDifferential=useDifferential,
                                                         differential_topN=differential_topN))

        return pd.DataFrame(data=results)

    def localization_Measurement(self, index, Measurement, cell_database=pd.DataFrame([]),
                                 RTT_precision_plus: float = 100.0, RTT_precision_minus: float = 100.0,
                                 unknownCell_punishment: float = 0.01, knownCell_bonus: float = 500.0,
                                 most_common: int = 2, lowCIDCount_punishment: bool = False, shiftRSRP: bool = False,
                                 useAbsolute: bool = True, useDifferential: bool = False, differential_topN: int = 2):
        """Positioning a measurement according to the given parameters.
        Steps:
            1. Selecting the potential Grid Cells
            2. Calculating the Similarity Score for every Grid Cell
            3. Keeping the top 50 Grid Cells with a Similarity Score >1
            4. Calculating the Error and returning the results for the estimated position calculation


        Args:
            index (int): The index/id of the measurement.
            Measurement (dictionary): The measurement, contans the position and the measured CIDs and RSRPs
            unknownCell_punishment (float, optional): The punishment of an unknown (not occured in the cell before) CID in the measurement. Defaults to 0.01.
            knownCell_bonus (float, optional): The bonus of a known CID (occured in the cell before). Defaults to 500.0.
            most_common (int, optional): The strongest CIDs in the measurement to use at the selection of the postential cells. Defaults to 2.
            lowCIDCount_punishment (bool, optional): The usage 3rd law is optional. If the offline measurements contain more CIDs than the online measurements than it's better not to use it.. Defaults to True.
            shiftRSRP (bool, optional): A test feature that shifts the Signal Strength values with the difference of the Serving Cell's Signal Strength and it's expected value. Defaults to False.

        Returns:
            result [dictionary]: A dictionary containing the basic attributes of the measurement (position, CID count) and the 50 most similar Grid Cells and their Similarity Score and Error
        """
        # * Initializing the result with the basic attributes
        result = {}
        result['id'] = index
        result['Lat'] = Measurement['Lat']
        result['Long'] = Measurement['Long']
        result['CID_count'] = len(getRSRPfrom_Measurement(Measurement, measurement_type=self.measurement_type))

        # * Selecting the potential Grid Cells based on the Grid's cellSelection and the measurement
        iCellList = [self.cellSelection[CID] for CID, point in
                     getPointsfrom_Measurement(Measurement, measurement_type=self.measurement_type).most_common(
                         most_common) if CID in self.cellSelection]
        if iCellList:
            iCells = set.union(*iCellList)
        else:
            iCells = list(self.grid.keys())

        measurementDistanceFromCellTower = -1
        result['Distance from Serving tower'] = measurementDistanceFromCellTower

        # * Calculating the Similarity Score for each potential Grid Cell and keeping the top 50
        CellScore = []
        for cell in iCells:
            p, unknownCells_number, knownCells_number = self.grid[cell].calculateSimilarityScore(Measurement,
                 unknownCell_punishment=unknownCell_punishment,
                 knownCell_bonus=knownCell_bonus,
                 lowCIDCount_punishment=lowCIDCount_punishment,
                 shiftRSRP=shiftRSRP,
                 useAbsolute=useAbsolute,
                 useDifferential=useDifferential,
                 differential_topN=differential_topN)
            if not np.isnan(p) and int(p) > 0:
                CellScore.append((int(p), cell))
        CellScore = heapq.nlargest(50, CellScore, key=lambda tup: tup[0])

        # * Calculating the errors for the most similar Grid Cells and storing the results
        lat_true, long_true = Measurement['Lat'] / 10 ** 6, Measurement['Long'] / 10 ** 6
        for idx, (p, cell) in enumerate(CellScore):
            distance = geopy.distance.distance((lat_true, long_true),
                                               (self.grid[cell].lat / 10 ** 6, self.grid[cell].long / 10 ** 6)).m
            result['Score_' + str(idx)] = p
            result['Error_' + str(idx)] = distance
            result['Cell_' + str(idx)] = cell
        return result

