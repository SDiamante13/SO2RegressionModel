from matplotlib import pyplot as plt
from scipy.stats.stats import pearsonr
from scipy import stats
import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf

class StatsModel:

    def readData(self, fileName):
        df = pd.read_csv(fileName)
        return df
    def createRegressionStats(self, independentVar, dependentVar):
        slope, intercept, r_value, p_value, std_err = stats.linregress(independentVar, dependentVar)
        print('R^2: ', r_value**2)
        print('slope: ', slope)
        print('intercept: ', intercept)
    def create_DV_IV_Plots(self, dataFrame):
        y_title_margin = 1.0
        fig1, ax1 = plt.subplots(ncols=3, nrows=1, squeeze=False, sharey = True)
        figManager = plt.get_current_fig_manager()
        figManager.window.state('zoomed') # Full Screen
        fig1.canvas.set_window_title('Figure 1')
        fig1.suptitle("Regression Fits for Good Variables", fontsize = 18, y = 0.036)
        ax1[0][0].set_title("SO2 vs Car Count", y = y_title_margin)
        ax1[0][1].set_title("SO2 vs Wind Velocity", y = y_title_margin)
        ax1[0][2].set_title("SO2 vs Temperature at 1 m", y = y_title_margin)

        # Using seaborn to produce regression plots for 3 important correlated independent variables
        sns.regplot(x='carCount', y='so2', data = dataFrame, ax= ax1[0][0])
        sns.regplot(x='windVelocity', y='so2', data = dataFrame, ax= ax1[0][1])
        sns.regplot(x='temperatureAtOne', y='so2', data = dataFrame, ax= ax1[0][2])

        # Set X and Y Axis
        ax1[0][0].set_ylim(3, 140)
        ax1[0][0].set_xlim(50, 4000) # carCount
        ax1[0][1].set_ylim(3, 140)
        ax1[0][1].set_xlim(0, 15) # windVelocity
        ax1[0][2].set_ylim(3, 140)
        ax1[0][2].set_xlim(-25, 20) # temperatureAtOne

        #axis margins
        plt.subplots_adjust(wspace=-1.51, hspace=1.5, left = -2)

        # display regression subplots
        fig1.tight_layout(pad=1.5, w_pad=0.1, h_pad=1.0)
        plt.show()
    def create_IV_IV_Plots(self, dataFrame):
        y_title_margin = 1.0
        fig2, ax2 = plt.subplots(ncols=3, nrows=1, squeeze=False)
        figManager = plt.get_current_fig_manager()
        figManager.window.state('zoomed') # Full Screen
        fig2.canvas.set_window_title('Figure 2')
        fig2.suptitle("Regression Fits IV-IV", fontsize = 18, y = 0.036)
        ax2[0][0].set_title("Wind Velocity vs. Car Count", y = y_title_margin) # x1 vs x2
        ax2[0][1].set_title("Temperature at 1m vs. Car Count", y = y_title_margin) # x1 vs x3
        ax2[0][2].set_title("Temperature 1m vs. Wind Velocity", y = y_title_margin) # x2 vs x3

        # Using seaborn to produce regression plots for 3 important correlated independent variables
        sns.regplot(x='carCount', y='windVelocity', data = dataFrame, ax= ax2[0][0])
        sns.regplot(x='carCount', y='temperatureAtOne', data = dataFrame, ax= ax2[0][1])
        sns.regplot(x='windVelocity', y='temperatureAtOne', data = dataFrame, ax= ax2[0][2])

        # Set X and Y Axis
        ax2[0][0].set_ylim(0, 15) # windVelocity
        ax2[0][0].set_xlim(50, 4000) # carCount
        ax2[0][1].set_ylim(-25, 20) # temperatureAtOne
        ax2[0][1].set_xlim(50, 4000) # carCount
        ax2[0][2].set_ylim(-25, 20) # temperatureAtOne
        ax2[0][2].set_xlim(0, 15) # windVelocity

        #axis margins
        plt.subplots_adjust(wspace=-1.51, hspace=1.5, left = -2)

        # display regression subplots
        fig2.tight_layout(pad=1.5, w_pad=0.1, h_pad=1.0)
        plt.show()
    def createRemovedVariablesPlot(self, dataFrame):
        y_title_margin = 1.0
        fig3, ax3 = plt.subplots(ncols=2, nrows=2)
        fig3.canvas.set_window_title('Figure 3')
        figManager = plt.get_current_fig_manager()
        figManager.window.state('zoomed') # Full Screen
        fig3.suptitle("Regression Fits of removed variables", fontsize = 20, y=1.0)
        ax3[0][0].set_title("SO2 vs Wind Direction", y=y_title_margin)
        ax3[0][1].set_title("SO2 vs Time Of Day", y = y_title_margin)
        ax3[1][0].set_title("SO2 vs Day Index", y = y_title_margin)
        ax3[1][1].set_title("SO2 vs Temperature at 30 m", y = y_title_margin)

        sns.regplot(x='windDirection', y='so2', data = dataFrame, ax= ax3[0][0])
        sns.regplot(x='timeOfDay', y='so2', data = dataFrame, ax= ax3[0][1])
        sns.regplot(x='dayIndex', y='so2', data = dataFrame, ax= ax3[1][0])
        sns.regplot(x='temperatureAtThirty', y='so2', data = dataFrame, ax= ax3[1][1])

        # Set X and Y Axis
        ax3[0][0].set_ylim(3, 100)
        ax3[0][0].set_xlim(0, 350) #windDirection
        ax3[0][1].set_ylim(3, 130)
        ax3[0][1].set_xlim(1, 24)  # timeOfDay
        ax3[1][0].set_ylim(3, 130)
        ax3[1][0].set_xlim(30, 200) # dayIndex
        ax3[1][1].set_ylim(3, 130)
        ax3[1][1].set_xlim(-25, 20) # temperatureAtThirty

        #axis margins
        plt.subplots_adjust(wspace=-1.51, hspace=1.5, left = -2)

        # display regression subplots
        fig3.tight_layout(pad=1.5, w_pad=0.1, h_pad=1.0)
        plt.show()
    def printRegressionStats(self, dataFrame):
        print('\nRegression Stats: SO2 vs. Car Count')
        statsModel.createRegressionStats(dataFrame['carCount'], dataFrame['so2'])

        print('\nRegression Stats: SO2 vs. Wind Velocity')
        statsModel.createRegressionStats(dataFrame['windVelocity'], dataFrame['so2'])

        print('\nRegression Stats: SO2 vs. Wind Direction')
        statsModel.createRegressionStats(dataFrame['windDirection'], dataFrame['so2'])

        print('\nRegression Stats: SO2 vs. Time Of Day')
        statsModel.createRegressionStats(dataFrame['timeOfDay'], dataFrame['so2'])

        print('\nRegression Stats: SO2 vs. Day Index')
        statsModel.createRegressionStats(dataFrame['dayIndex'], dataFrame['so2'])

        print('\nRegression Stats: SO2 vs. Temperature 1m')
        statsModel.createRegressionStats(dataFrame['temperatureAtOne'], dataFrame['so2'])

        print('\nRegression Stats: SO2 vs. Temperature 30m')
        statsModel.createRegressionStats(dataFrame['temperatureAtThirty'], dataFrame['so2'])
    def calculateCorrelationCoefficients_DV_vs_IVs(self, dataFrame):
        pearsonr_coefficient, p_value = pearsonr(dataFrame['carCount'], dataFrame['so2'])
        print('\nCorrelation between SO2 and Car Count')
        print('PearsonR Coefficient Value: %0.3f' % (pearsonr_coefficient))

        pearsonr_coefficient, p_value = pearsonr(dataFrame['windVelocity'], dataFrame['so2'])
        print('Correlation between SO2 and Wind Velocity')
        print('PearsonR Coefficient Value: %0.3f' % (pearsonr_coefficient))

        pearsonr_coefficient, p_value = pearsonr(dataFrame['temperatureAtOne'], dataFrame['so2'])
        print('Correlation between SO2 and Temperature 1m')
        print('PearsonR Coefficient Value: %0.3f' % (pearsonr_coefficient))
    def calculateCorrelationCoefficients_IVs_vs_IVs(self, dataFrame):
        print('\n IV vs. IV Correlation Values')
        pearsonr_coefficient, p_value = pearsonr(dataFrame['carCount'], dataFrame['windVelocity'])
        print('Correlation between Wind Velocity and Car Count')
        print('PearsonR Coefficient Value: %0.3f' % (pearsonr_coefficient))

        pearsonr_coefficient, p_value = pearsonr(dataFrame['carCount'], dataFrame['temperatureAtOne'])
        print('Correlation between Temperature 1m and Car Count')
        print('PearsonR Coefficient Value: %0.3f' % (pearsonr_coefficient))

        pearsonr_coefficient, p_value = pearsonr(dataFrame['windVelocity'], dataFrame['temperatureAtOne'])
        print('Correlation between Temperature 1m and Wind Velocity')
        print('PearsonR Coefficient Value: %0.3f \n' % (pearsonr_coefficient))
    def runMultipleRegressionAnalysis(self, trainData3F):
        #drop 100 rows from SO2 to make it make testData
        #sulfurDioxide = trainData3F['so2']
        #sulfurDioxide = sulfurDioxide[:200]
        regressor = LinearRegression()
        linearModel = smf.ols(formula = 'so2 ~ carCount + windVelocity  + temperatureAtOne + temperatureAtThirty', data = trainData3F).fit()
        print(linearModel.params)
        print(linearModel.summary())
    def runPrediction(self, trainDataWithoutDep, trainData3F, testData3F):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(trainDataWithoutDep, trainData3F['so2'], random_state=1)
        # Instantiate model
        lm2 = LinearRegression()
        # Fit Model
        lm2.fit(X_train, y_train)
        # Predict
        so2_pred = lm2.predict(testData3F)
        return so2_pred
    def outputToFile(self, outputFileName, predictionValuesSO2):
        # Output SO2 predictions to PREDICTED_DATA.txt
        np.savetxt(outputFileName, predictionValuesSO2, delimiter=",", fmt = '%0.4f')
    def printDataFrameStats(self, dataFrame):
        with pd.option_context("display.max_rows", 200, "display.max_columns", 3):
            print(dataFrame.head(200).describe())

#   Main
#   This program will read in training data for a SO2 gas
#   It will run a multiple regression analysis against 3 independent variables
#   It will output prediction values of the SO2 gas content based on the prediction model
statsModel = StatsModel()
trainDataFileName = "C:/Users/steel/Atom_Python/SO2RegressionModel/TRAINING_DATA.txt"
dataFrame = statsModel.readData(trainDataFileName)

#Renaming dataFrame variables
dataFrame.columns = ['so2', 'carCount', 'windVelocity', 'windDirection', 'timeOfDay', 'dayIndex', 'temperatureAtOne', 'temperatureAtThirty']

#Regression Plots for Dependent Variable vs Independent Variables
statsModel.create_DV_IV_Plots(dataFrame)
#IV - IV Scatter Plots
statsModel.create_IV_IV_Plots(dataFrame)
#Last Figure of remaining 4 variables that I have removed from the regression
statsModel.createRemovedVariablesPlot(dataFrame)
#Regression Stats
statsModel.printRegressionStats(dataFrame)
#Correlation DV vs. IVs
statsModel.calculateCorrelationCoefficients_DV_vs_IVs(dataFrame)
#Correlation IVs vs. IVs
statsModel.calculateCorrelationCoefficients_IVs_vs_IVs(dataFrame)
#Getting rid of some Variables and dropping 100 rows so that trainData and testData have same # of rows
trainData3F = dataFrame.drop(['windDirection', 'timeOfDay', 'dayIndex'], axis = 1)
trainData3F = trainData3F[:200]
#Let's also rename variables without using '.'
trainData3F.columns = ['so2', 'carCount', 'windVelocity', 'temperatureAtOne', 'temperatureAtThirty']
#Reading in testdata
testDataFileName = "C:/Users/steel/Atom_Python/SO2RegressionModel/TESTING_DATA.txt"
testData= statsModel.readData(testDataFileName)
testData3F = testData.drop(['wind.direction', 'time.of.day', 'day.index'], axis = 1)
outputFileName = "C:/Users/steel/Atom_Python/SO2RegressionModel/PREDICTED_DATA.txt"

#Multiple Regression Analysis
statsModel.runMultipleRegressionAnalysis(trainData3F)
#Prediction Testing
trainDataWithoutDep = trainData3F.drop(['so2'], axis = 1)
predictionValuesSO2 = statsModel.runPrediction(trainDataWithoutDep, trainData3F, testData3F)
statsModel.outputToFile(outputFileName, predictionValuesSO2)
statsModel.printDataFrameStats(trainData3F)

'''
END
'''
