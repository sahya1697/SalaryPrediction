# SalaryPrediction
Engineering Salary Prediction: Regression vs Classification

<b> Data preprocessing </b> is a crucial step in data analysis and machine learning. The steps included are:

Cleaning and Handling Missing Data:

Identify and handle missing values (e.g., impute with mean, median, or mode).
Remove duplicate records.
Address outliers that might skew your analysis.

Feature Transformation:

Standardize or normalize features to bring them to a common scale.
Encode categorical variables (e.g., one-hot encoding) for machine learning algorithms.
Create new features (e.g., combining existing ones).

Data Reduction:

Reduce dimensionality (e.g., using PCA or feature selection techniques).
Remove irrelevant or redundant features.

Data Formatting:

Ensure consistent data types (e.g., converting dates to a standard format).
Handle text data (cleaning, tokenization, stemming, etc.).



<b>REGRESSION</b>

1.linear regression

Linear regression is a statistical model that estimates the linear relationship between a scalar response (often denoted as (y)) and one or more explanatory variables (also known as independent variables, often denoted as (x))

Parameters: fit_interceptbool, default=True Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).

copy_Xbool, default=True If True, X will be copied; else, it may be overwritten.

n_jobsint, default=None The number of jobs to use for the computation. This will only provide speedup in case of sufficiently large problems, that is if firstly n_targets > 1 and secondly X is sparse or if positive is set to True. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.

positivebool, default=False When set to True, forces the coefficients to be positive. This option is only supported for dense arrays.

2.knn regression

K-Nearest Neighbors (KNN) regression is a non-parametric algorithm that predicts the value of a target variable based on the average (or weighted average) of its (k) nearest neighbors in the feature space. It’s a simple yet effective method for regression tasks.

Parameters: n_neighborsint, default=5 Number of neighbors to use by default for kneighbors queries.

weights{‘uniform’, ‘distance’}, callable or None, default=’uniform’ Weight function used in prediction. Possible values:

‘uniform’ : uniform weights. All points in each neighborhood are weighted equally.

‘distance’ : weight points by the inverse of their distance. in this case, closer neighbors of a query point will have a greater influence than neighbors which are further away.

[callable] : a user-defined function which accepts an array of distances, and returns an array of the same shape containing the weights.

Uniform weights are used by default.

algorithm{‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’}, default=’auto’
Algorithm used to compute the nearest neighbors:

‘ball_tree’ will use BallTree

‘kd_tree’ will use KDTree

‘brute’ will use a brute-force search.

‘auto’ will attempt to decide the most appropriate algorithm based on the values passed to fit method.

Note: fitting on sparse input will override the setting of this parameter, using brute force.

leaf_sizeint, default=30
Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree. The optimal value depends on the nature of the problem.

pfloat, default=2
Power parameter for the Minkowski metric. When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

metricstr, DistanceMetric object or callable, default=’minkowski’
Metric to use for distance computation. Default is “minkowski”, which results in the standard Euclidean distance when p = 2. See the documentation of scipy.spatial.distance and the metrics listed in distance_metrics for valid metric values.

If metric is “precomputed”, X is assumed to be a distance matrix and must be square during fit. X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors.

If metric is a callable function, it takes two arrays representing 1D vectors as inputs and must return one value indicating the distance between those vectors. This works for Scipy’s metrics, but is less efficient than passing the metric name as a string.

If metric is a DistanceMetric object, it will be passed directly to the underlying computation routines.

metric_paramsdict, default=None
Additional keyword arguments for the metric function.

n_jobsint, default=None
The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details. Doesn’t affect fit method.

3.Extreme Gradient boost regression

XGBoost (Extreme Gradient Boosting) is a powerful gradient boosting algorithm for regression tasks. It combines decision trees and gradient boosting to achieve high predictive accuracy.

parameters: n_estimators The number of boosting stages that will be performed. Later we will plot deviance against boosting iterations.

max_depth Limits the number of nodes in the tree. The best value depends on the interaction of the input variables.

min_samples_split The minimum number of samples required to split an internal node.

4.Random forest regression*

random forest regressor combines multiple decision tree regressors to improve predictions by averaging their outputs. It’s an effective ensemble technique for regression tasks.

parameters:

n_estimators Number of trees in the forest. max_depth Maximum depth of each tree. min_samples_split Minimum samples required to split an internal node. min_samples_leaf Minimum samples required at a leaf node.

5.Ridge regression

Ridge regression is a statistical regularization technique used in linear regression models.

Parameters: alphafloat or array-like of shape (n_targets,) Constant that multiplies the L2 term, controlling regularization strength. alpha must be a non-negative float i.e. in [0, inf).

sample_weightfloat or array-like of shape (n_samples,), default=None Individual weights for each sample. If given a float, every sample will have the same weight. If sample_weight is not None and solver=’auto’, the solver will be set to ‘cholesky’.

solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’ Solver to use in the computational routines. All solvers except ‘svd’ support both dense and sparse data. However, only ‘lsqr’, ‘sag’, ‘sparse_cg’, and ‘lbfgs’ support sparse input when fit_intercept is True.

max_iterint, default=None Maximum number of iterations for conjugate gradient solver. For the ‘sparse_cg’ and ‘lsqr’ solvers, the default value is determined by scipy.sparse.linalg. For ‘sag’ and saga solver, the default value is 1000. For ‘lbfgs’ solver, the default value is 15000.

tolfloat, default=1e-4 Precision of the solution. Note that tol has no effect for solvers ‘svd’ and ‘cholesky’.

verboseint, default=0 Verbosity level. Setting verbose > 0 will display additional information depending on the solver used.

positivebool, default=False When set to True, forces the coefficients to be positive. Only ‘lbfgs’ solver is supported in this case.

random_stateint, RandomState instance, default=None Used when solver == ‘sag’ or ‘saga’ to shuffle the data. See Glossary for details.

return_n_iterbool, default=False If True, the method also returns n_iter, the actual number of iteration performed by the solver.

return_interceptbool, default=False If True and if X is sparse, the method also returns the intercept, and the solver is automatically changed to ‘sag’. This is only a temporary fix for fitting the intercept with sparse data. For dense data, use sklearn.linear_model._preprocess_data before your regression.

check_inputbool, default=True If False, the input arrays X and y will not be checked.

6.Neural network

A neural network regressor is a machine learning model that uses interconnected layers of artificial neurons to predict continuous numeric values (regression tasks). It learns from data by adjusting weights during training.




<b> Evaluation Matrics </b>

1.Mean Absolute Error (MAE): This metric calculates the average absolute difference between the predicted values and the actual target values. It provides a straightforward measure of prediction accuracy. The formula for MAE is:

MAE=n1​i=1∑n​∣yi​−y^​i​∣

where:

(n) is the number of data points. (y_i) represents the actual target value. (\hat{y}_i) represents the predicted value.

2.Mean Squared Error (MSE): MSE computes the average squared difference between predicted and actual values. It penalizes larger errors more heavily. The formula for MSE is: MSE=n1​i=1∑n​(yi​−y^​i​)2

3.Root Mean Squared Error (RMSE): RMSE is the square root of MSE. It provides a measure of the typical error magnitude. The formula for RMSE is: RMSE=n1​i=1∑n​(yi​−y^​i​)2​

4.R-squared (Coefficient of Determination): R-squared represents the proportion of variance in the target variable explained by the model. It ranges from 0 to 1, with higher values indicating better fit. An R-squared close to 1 suggests a good model fit.
