# General Imports
import pandas as pd
import numpy as np
import statistics as s
import feather
import pdfkit as pdf
import itertools

# Sklearn imports
from sklearn.linear_model import LinearRegression, Lasso, LassoCV, Ridge, TheilSenRegressor, RANSACRegressor, \
    HuberRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
from scipy import stats

# Matplotlib imports
from matplotlib.legend_handler import HandlerLine2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sea


# HELPER FUNCTIONS FOR Q2

def convert_dataframe_into_rdata(df, name):
    path = name
    feather.write_dataframe(df, path)


# Helper function to keep track of probabilities of picking 6-16 for each team
def updateDeterministicPicks(currentProbList, deterministicPickProbs, currentConditionalProb):
    # Iterate through remaining 11 teams
    for i in range(11):
        # Pick the worst team out of the remaining teams we have and update its
        # probability of receiving i+5th pick
        worst_team = np.nonzero(currentProbList)[0]
        curr_team = min(worst_team)
        # update for this scenario
        deterministicPickProbs[i][curr_team] += currentConditionalProb

        # remove this team from remaining teams, so that we have a new worst
        # team
        currentProbList[curr_team] = 0
    return deterministicPickProbs


# Helper function to calculate conditional probabilities
def calculateRemainingProbabilities(initial_probabilities, curr_team,
                                    secondPick, *args):
    # Second Pick is a unique case because the initial_probabilities are used
    # in calculating conditional probabilities

    if secondPick:
        # Remove the probability of the current team
        remainingProbAfterPick = [initial_probabilities[index] if index != curr_team else 0. for index in
                                  range(len(initial_probabilities))]
        # Recalculate new conditional probabilities for the remaining teams
        conditionalProbs = [initial_probabilities[curr_team] * x / sum(remainingProbAfterPick) for x in
                            remainingProbAfterPick]
        # Add the calculated probabilities to the respective probability list
        return remainingProbAfterPick, conditionalProbs

    else:
        # Our conditional probability list is not equal to initial
        # probabilities.
        updated_conditional_probabilities = args[0]
        remainingProbAfterPick = [initial_probabilities[index] if index != curr_team else 0. for index in
                                  range(len(initial_probabilities))]
        conditionalProbs = [updated_conditional_probabilities[curr_team] * x / sum(remainingProbAfterPick) for x in
                            remainingProbAfterPick]
        return remainingProbAfterPick, conditionalProbs


# HELPER FUNCTIONS FOR Q1

# Helper function to print coefficients and intercepts of a linear
# regression model

def pretty_print_linear(coefs, intercept, names=None, sort=False):
    if names == None:
        names = ["X%s" % x for x in range(1, len(coefs[0]))]
    coefs = coefs[0]
    del coefs[0]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))

    linear_formula = " + ".join("%s * %s" % (np.round(coef, 3), name)
                                for coef, name in lst)
    linear_formula = str(np.round(intercept, 3)) + " + " + linear_formula
    return linear_formula


def plotRegressionLine(x_df_train, y_df_train, x_df_test, y_pred, y_df_test):
    plt.scatter(y_pred, y_df_test['y'], s=2)
    plt.plot(y_pred, y_pred, color='red')
    plt.show()


def metrics(y, rsquared, X):
    adj_r_squared = 1 - (1 - rsquared) * (len(y) - 1) / \
                    (len(y) - X.shape[1] - 1)

    return adj_r_squared


# Helper function I used to test different regression models like Lasso,
# Ridge, TheilSenRegressor, RANSACRegressor,
def testRegressionModels(features, labels):
    for i in range(10):
        print("Random seed %s" % i)
        np.random.seed(seed=i)
        x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2,
                                                            random_state=np.random.seed(seed=i))
        lr = LinearRegression()
        lr.fit(x_train, y_train)

        print("Linear model: {}".format(
            pretty_print_linear(lr.coef_.tolist(), lr.intercept_[0])))

        ridge = Ridge(alpha=10)

        ridge.fit(x_train, y_train)
        print("Ridge model: {}".format(pretty_print_linear(
            ridge.coef_.tolist(), ridge.intercept_[0])))


# Class for filling up 2019 hypothetical NBA Draft probabilities
class NFLDraftProbabilitySimulator(object):

    def __init__(self, initial_probabilities):

        self.initial_probabilities = initial_probabilities

        # These list of lists will hold conditional probabilities for picks 2-5
        # for each team for every possible permutation
        self.secondpickprobslist = []
        self.thirdpickprobslist = []
        self.fourthpickprobslist = []
        self.fifthpickprobslist = []

        # The below matrix will hold cumulative probabilities between picks 6-16, which
        # are fixed after we go through picks 2-5
        self.deterministicPickProbs = [[0] * 16 for _ in range(11)]

        self.teams = list(range(1, 17))

    def createDraftProbabilities(self):
        # Temp Variables
        curr_i = 100
        curr_j = 100
        curr_k = 100
        curr_l = 100
        for i, j, k, l, m in itertools.permutations(range(16), 5):

            pick_order = [i, j, k, l, m]
            print("Current pick order between 1-5 is {}".format(pick_order))

            # Calculate conditional probabilities for remaining teams after Team i gets its pick
            # Also delete Team i's probability for future calculations
            remainingProbAfterFirst, conditionalProbs = calculateRemainingProbabilities(
                self.initial_probabilities, i, True)

            # This if statement is necessary because we want to capture cond.
            # probability of each permutation once
            if i != curr_i:
                self.secondpickprobslist.append(conditionalProbs)
                curr_i = i

            # Pick number 3
            remainingProbAfterSecond, conditionalProbs2 = calculateRemainingProbabilities(
                remainingProbAfterFirst, j, False, conditionalProbs)

            if j != curr_j:
                self.thirdpickprobslist.append(conditionalProbs2)
                curr_j = j
            # Pick number 4
            remainingProbAfterThird, conditionalProbs3 = calculateRemainingProbabilities(
                remainingProbAfterSecond, k, False, conditionalProbs2)

            if k != curr_k:
                self.fourthpickprobslist.append(conditionalProbs3)
                curr_k = k
            # Pick number 5
            remainingProbAfterFourth, conditionalProbs4 = calculateRemainingProbabilities(
                remainingProbAfterThird, l, False, conditionalProbs3)

            if l != curr_l:
                self.fifthpickprobslist.append(conditionalProbs4)
                curr_l = l

            # Remove the fourth team from our conditional probability list
            remainingProbAfterFifth = [remainingProbAfterFourth[index] if index != m else 0. for index
                                       in range(len(remainingProbAfterFourth))]
            # Based on the current order, get the remaining cond. probability and conditional probability of all teams
            # and calculate probabilities for picks 6-16
            self.deterministicPickProbs = updateDeterministicPicks(remainingProbAfterFifth, self.deterministicPickProbs,
                                                                   conditionalProbs4[m])

    def createDraftPDF(self):

        # Sum up accross all possible scenarios for each teams conditional
        # probability for picks 2-5
        secondpickProbability = [sum(elts)
                                 for elts in zip(*self.secondpickprobslist)]

        thirdpickProbability = [sum(elts)
                                for elts in zip(*self.thirdpickprobslist)]

        fourthpickprobability = [sum(elts)
                                 for elts in zip(*self.fourthpickprobslist)]

        fifthpickprobability = [sum(elts)
                                for elts in zip(*self.fifthpickprobslist)]
        df = pd.DataFrame()
        df['Team Number'] = self.teams
        df['First Pick'] = self.initial_probabilities
        df['Second Pick'] = secondpickProbability
        df['Third Pick'] = thirdpickProbability
        df['Fourth Pick'] = fourthpickprobability
        df['Fifth Pick'] = fifthpickprobability
        df['Sixth Pick'] = self.deterministicPickProbs[0]
        df['Seventh Pick'] = self.deterministicPickProbs[1]
        df['Eighth Pick'] = self.deterministicPickProbs[2]
        df['Ninth Pick'] = self.deterministicPickProbs[3]
        df['Tenth Pick'] = self.deterministicPickProbs[4]
        df['Eleventh Pick'] = self.deterministicPickProbs[5]
        df['Twelfth Pick'] = self.deterministicPickProbs[6]
        df['Thirteenth Pick'] = self.deterministicPickProbs[7]
        df['Fourteenth Pick'] = self.deterministicPickProbs[8]
        df['Fifteenth Pick'] = self.deterministicPickProbs[9]
        df['Sixteenth Pick'] = self.deterministicPickProbs[10]
        # drop indexes and do 6 decimals
        df.reset_index(drop=True, inplace=True)
        df.round(6)
        # Convert to pdf
        df.to_html('test.html')
        PdfFilename = 'NBA2019DRAFT.pdf'
        pdf.from_file('test.html', PdfFilename)


# Class to visualize, develop, train,test and create predictions for a
# predictive model
class PredictiveModel(object):

    def __init__(self, data, preddata):
        self.trainData = pd.read_csv(data, float_precision='round_trip')
        predData = pd.read_csv(preddata, float_precision='round_trip')
        predData.drop(['ID'], axis=1, inplace=True)
        feature1 = self.trainData[['x1']].values
        feature2 = self.trainData[['x2']].values
        labels = self.trainData[['y']].values
        features = np.hstack((feature1, feature2))
        self.pred_features = np.hstack(
            (predData[['x1']].values, predData[['x2']].values))

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(features, labels, test_size=0.35,
                                                                                random_state=42)

    # Visualize relationship between variables, calculate corr. matrix and
    # linear
    def visualizeData(self, degree):
        # Scatter plots between 2 variables
        plt.scatter(self.trainData['x1'], self.trainData[
            'y'], color='red', s=2)
        plt.title('x1 Vs Target', fontsize=14)
        plt.xlabel('x1 ', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.grid(True)
        plt.show()

        plt.scatter(self.trainData['x2'], self.trainData[
            'y'], color='red', s=2)
        plt.title('x2 Vs Target', fontsize=14)
        plt.xlabel('x2 ', fontsize=14)
        plt.ylabel('y', fontsize=14)
        plt.grid(True)
        plt.show()

        plt.scatter(self.trainData['x2'], self.trainData[
            'x1'], color='red', s=2)
        plt.title('x2 Vs x1', fontsize=14)
        plt.xlabel('x2 ', fontsize=14)
        plt.ylabel('x1', fontsize=14)
        plt.grid(True)
        plt.show()

        # Regression Line of a certain order
        sea.lmplot(x="x1", y="y", data=self.trainData,
                   scatter_kws={"s": 1}, ci=None, order=degree)
        plt.show()
        sea.lmplot(x="x2", y="y", data=self.trainData,
                   scatter_kws={"s": 1}, ci=None, order=degree)
        plt.show()
        print("Correlation table {}".format(self.trainData.corr()))

    # Visualize data 3d
    def visualizeData3d(self):
        fig1 = plt.figure(figsize=(10, 10))
        ax = fig1.gca(projection='3d')
        ax.scatter(self.trainData['x1'], self.trainData[
            'x2'], self.trainData['y'])
        ax.set_title("x1, x2, y")
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_zlabel('Y')
        plt.show()

    # Visualize linear regression plane and scatter the residuals above and
    # below the regression plane
    def visualizeHyperPlane(self):
        # Convert everything back to data frames
        x_df_train = pd.DataFrame(self.x_train)
        y_df_train = pd.DataFrame(self.y_train)
        x_df_train.columns = ['x1', 'x2']
        y_df_train.columns = ['y']

        x_df_test = pd.DataFrame(self.x_test)
        y_df_test = pd.DataFrame(self.y_test)
        x_df_test.columns = ['x1', 'x2']
        y_df_test.columns = ['y']

        # Run Linear Regression
        regr = LinearRegression()
        regr.fit(x_df_train, y_df_train)
        # print('Intercept: \n', regr.intercept_)
        # print('Coefficients: \n', regr.coef_[0])
        coefficients = regr.coef_[0]

        x_surf = np.linspace(x_df_train.x1.min(), x_df_train.x1.max(), 100)
        y_surf = np.linspace(x_df_train.x2.min(), x_df_train.x2.max(), 100)
        x_surf, y_surf = np.meshgrid(x_surf, y_surf)

        # Z surface is the hyperplane generated by linear regression intercept
        # and coefs.
        z_surf = regr.intercept_[0] + coefficients[0] * \
                 x_surf + coefficients[1] * y_surf

        # Plot regression line
        # plotRegressionLine(x_df_train, y_df_train, x_df_test, regr.predict(x_df_test), y_df_test)

        fig1 = plt.figure(figsize=(10, 10))
        ax = fig1.gca(projection='3d')
        ax.plot_surface(x_surf, y_surf, z_surf,
                        cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

        # Calculate residuals
        resid = y_df_test - regr.predict(x_df_test)

        above_resid = resid['y'] >= 0
        below_resid = resid['y'] < 0

        # If the residual is positive, color the point white
        ax.scatter(x_df_test[above_resid].x1, x_df_test[above_resid].x2, y_df_test[above_resid].y, color='black',
                   alpha=1.0,
                   facecolor='white')

        # If the residual is negative, color the point black
        ax.scatter(x_df_test[below_resid].x1, x_df_test[
            below_resid].x2, y_df_test[below_resid].y, color='black')

        # Show the plot
        plt.xlabel('x1')
        plt.ylabel('x2')
        ax.set_zlabel('Y')
        ax.axis('equal')
        ax.axis('tight')
        plt.show()

    # Test RMSE, R-Squared and Explained Variance with different degree
    # polynomials + cross validation
    def testPolynomialDegrees(self):
        # Variables
        degrees = list(range(1, 10))
        train_rmse = []
        test_rmse = []
        train_rsquared = []
        test_rsquared = []
        train_explained_var = []
        test_explained_var = []

        # Get a summary of the linear model
        x_train_ols = sm.add_constant(self.x_train)  # adding a constant
        model2 = sm.OLS(self.y_train, x_train_ols).fit()
        print(model2.summary())

        for degree in degrees:
            model = make_pipeline(PolynomialFeatures(degree, interaction_only=False, include_bias=True),
                                  LinearRegression())
            model.fit(self.x_train, self.y_train)
            print("Current Degree is {}".format(degree))

            y_train_pred = model.predict(self.x_train)
            y_test_pred = model.predict(self.x_test)

            train_rsquared.append(r2_score(self.y_train, y_train_pred))
            test_rsquared.append(r2_score(self.y_test, y_test_pred))
            train_explained_var.append(
                explained_variance_score(self.y_train, y_train_pred))
            test_explained_var.append(
                explained_variance_score(self.y_test, y_test_pred))

            print("Linear model:", pretty_print_linear(model.named_steps['linearregression'].coef_.tolist(),
                                                       model.named_steps['linearregression'].intercept_[0]))

            train_scores = cross_val_score(
                model, self.x_train, self.y_train, scoring="neg_mean_squared_error", cv=10)
            test_scores = cross_val_score(
                model, self.x_test, self.y_test, scoring="neg_mean_squared_error", cv=10)

            train_rmse.append(s.mean(np.sqrt(-train_scores)))
            test_rmse.append(s.mean(np.sqrt(-test_scores)))

            print("Regression Score is {}".format(model.score(self.x_test, self.y_test)))
            print("Cross Validated (cv=10) RMSE Train scores = {}".format(str(s.mean(np.sqrt(-train_scores)))))
            print("Cross Validated RMSE (cv=10) Test scores = {}\n".format((s.mean(np.sqrt(-test_scores)))))

        line1, = plt.plot(degrees, train_rmse, 'b', label="Train RMSE")
        line2, = plt.plot(degrees, test_rmse, 'r', label="Test RMSE")
        line3, = plt.plot(degrees, train_rsquared, 'g', label="Train R^2")
        line4, = plt.plot(degrees, test_rsquared, 'black', label="Test R^2")
        line5, = plt.plot(degrees, test_explained_var,
                          'purple', label="Explained Variance Score")

        plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
        plt.xlabel("Degree")
        # print("Degrees are {}".format(degrees))
        # print("Train R Sqaured values are {}".format(train_rsquared))
        # print("Test R Sqaured values are {}".format(test_rsquared))
        # print("Train RMSE's are {}".format(train_rmse))
        # print("Test RMSE's are {}".format(test_rmse))
        # print("Explained Variance scores are {}".format(test_explained_var))
        # plt.show()
        plt.pause(0.1)

    def trainChosenModel(self, degree):
        print("Training the model with Polynomial degree 5")
        polynomial_features = PolynomialFeatures(degree=degree)
        x_poly = polynomial_features.fit_transform(self.x_train)
        x_poly_test = polynomial_features.fit_transform(self.x_test)

        x_pred = polynomial_features.fit_transform(self.pred_features)
        model = LinearRegression()
        model.fit(x_poly, self.y_train)

        train_scores = cross_val_score(
            model, x_poly, self.y_train, scoring="neg_mean_squared_error", cv=10)
        test_scores = cross_val_score(
            model, x_poly_test, self.y_test, scoring="neg_mean_squared_error", cv=10)

        print("Train RMSE {}".format(
            np.sqrt(mean_squared_error(self.y_train, model.predict(x_poly)))))
        print("Test RMSE {}".format(
            np.sqrt(mean_squared_error(self.y_test, model.predict(x_poly_test)))))
        print("Cross Validated Train RMSE with (cv=10) {}".format(
            s.mean(np.sqrt(-train_scores))))
        print("Cross Validated Test RMSE with (cv=10) {}".format(
            s.mean(np.sqrt(-test_scores))))
        print("Train R Squared Score {}".format(
            r2_score(self.y_train, model.predict(x_poly))))
        print("Test R Squared Score {}".format(
            r2_score(self.y_test, model.predict(x_poly_test))))

        y_pred = model.predict(x_pred)
        y_pred_df = pd.DataFrame(data=y_pred)
        y_pred_df.columns = ['y']
        y_pred_df.index.names = ['id']
        y_pred_df.index += 1

        y_pred_df.reset_index().to_csv('TestDataPredictions.csv', index=False, header=True, decimal='.', sep=',',
                                       encoding='utf-16')

        print("Statistics of our predictions {}".format(stats.describe(y_pred)))


def main():
    print("Running the main function")
    print("NBA Draft Probability Question")
    initial_probabilities = [.114, .113, .112, .111, .099, .089, .079, .069, .059, .049, .039, .029, .019, .009, .006,
                             .004]
    nbadraft_probabilities = NFLDraftProbabilitySimulator(initial_probabilities)
    # Calculate conditional probabilities
    nbadraft_probabilities.createDraftProbabilities()
    # Convert it to pdf
    nbadraft_probabilities.createDraftPDF()

    # print("Predictive Model Question")
    model = PredictiveModel("PredictiveModelingAssessmentData.csv", "TestData.csv")
    # Change these variables to True to visualize data 3d and with regression
    # hyperplane and to run Polynomial degree hyperparameter tuning
    visualize = True
    test_degrees = False
    if visualize:
        # visualize relationship betweeen variables and draw linear regression
        # of a given degree polynomial.
        model.visualizeData(degree=5)

        # Visualize Data 3d
        model.visualizeData3d()

        # Visualize HyperPlane of Linear Regression
        model.visualizeHyperPlane()

    # Decide on what Polynomial Degree fits the data best without overfitting.
    # I also tried Lasso and Ridge Regression
    if test_degrees:
        model.testPolynomialDegrees()

    # Train and Test the Polynomial Linear Regression before creating
    # predictions and saving in a csv file
    # model.trainChosenModel(degree=5)


if __name__ == '__main__':
    main()
