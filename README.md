Download Link: https://assignmentchef.com/product/solved-comp135-assignment-2
<br>
You have been given a data set (data.csv) consisting of some synthetically generated (input, output) (<em>x,y</em>) pairs, each represented by a single floating point value. You will write code that explores this data, using a number of different linear regression models.

<ol>

 <li>You fill first fit a sequence of models, starting from a simple (degree-1) linear regression, and then continuing for data that is augmented using an increasing sequence of higher-degree polynomials.

  <ul>

   <li>Fill in the test_polynomials This function should take in a list of positive integer values, corresponding to different degrees for polynomial models (so if the degree is 1, we get a purely linear model, if it is 2, we first transform the data so it is degree-2, and so on). Each model should be fit to the entire data-set, and then the model should be used to predict outputs for that data-set.</li>

  </ul></li>

</ol>

The function will return two lists. One will consist of the predictions, i.e., it will be a list of length <em>P</em>, where <em>P </em>is the number of different polynomial degrees, and each element of the list consists of the <em>N </em>output values, where <em>N </em>is the size of the overall data-set). The other will consist of the <em>P </em>error values for the models, where error is calculated using the mean squared error metric (MSE).

Once the function is completed, it should be called using the sequence of degrees <em>d </em>∈ {1<em>,</em>2<em>,</em>3<em>,</em>4<em>,</em>5<em>,</em>6<em>,</em>10<em>,</em>11<em>,</em>12}. Once the function returns its two lists, the plot_predictions function (already written for you) will produce a plot containing each of the models’ predictions, along with their MSE values (found in the title area of each subplot).

<ul>

 <li>Discuss the plotted results. What do they show? What is the best model, based upon MSE? What models do particularly poorly? What does this tell you?</li>

</ul>

<ol start="2">

 <li>You will now consider the same sequence of degrees as in the previous question. Instead of simply fitting the entire data-set, however, you will do 5-fold cross-validation, to allow an estimate of when the various models may be over-fitting to the data.</li>

</ol>

We have supplied a make_folds function. This function takes a positive integer value, <em>k</em>, and divides the data-set (both <em>x </em>and <em>y </em>values) into <em>k </em>distinct sub-parts, where each part consists of the same number of elements (we will assume for the sake of this assignment that the data-set size divides evenly by the value <em>k</em>). The folds are returned in the form of two lists (for <em>x </em>and <em>y</em>), each of length <em>k</em>, where each element of a list consists of some consecutive sub-sequence of the original data, and each data-element is a member of exactly one fold (so that printing the contents of the <em>k </em>folds in order would give us back the exact same data-set). You will use this function in the rest of your code.

<ul>

 <li>For each of the polynomial degrees previously considered, perform 5-fold crossvalidation on the data. That is, for each degree, your code should build 5 separate models, training each time on a different 4<em>/</em>5 of the data and testing on the remaining 1<em>/</em> You should use the make_folds function that we provided to split the data into the 5 parts—it is then up to you to re-combine parts as needed in each iteration of the cross-validation.</li>

</ul>

You should keep track of the average training and testing error (MSE) seen over the 5 models, for each degree. You will then plot these results. This should consist of a single plot with:

<ul>

 <li>The distinct polynomial degrees along the <em>x</em>-axis, and average MSE along the <em>y</em>-axis.</li>

 <li>Average error plotted as two separate lines, one for training data and one for testing data; the lines should be distinguishable, by color or other features.</li>

 <li>Proper labels on all axes, an explanatory title, and a legend making clear which line is which.</li>

</ul>

<strong>Also</strong>: Following the plot, there should be tabular text print-out of the data as well; this can be in any clear format.

<ul>

 <li>Discuss the plotted results. What do they show? Where do we see the best results? Where is their underfitting, and why do you say that? Where is there overfitting, and why do you say that?</li>

</ul>

<ol start="3">

 <li>You will now examine ways in which a basic linear/polynomial model, of a fixed degree, can be adjusted to help avoid overfitting.</li>

</ol>

One way to do this is via <em>ridge regression</em>. In this technique, rather than seek only to minimize error on the data-set, the regression model seeks to balance minimal error with weights that are smaller in magnitude (positive or negative)—that is, weights that are closer to 0. By doing this, there is often less chance that the model over-fits to one particular feature that is more characteristic of its training data than is representative of the overall truth.

Discussion of ridge regression can be found in James et al. (section 6.2.1, pp. 215–219) [<a href="https://static1.squarespace.com/static/5ff2adbe3fe4fe33db902812/t/6009dd9fa7bc363aa822d2c7/1611259312432/ISLR+Seventh+Printing.pdf">link</a><a href="https://static1.squarespace.com/static/5ff2adbe3fe4fe33db902812/t/6009dd9fa7bc363aa822d2c7/1611259312432/ISLR+Seventh+Printing.pdf">]</a>

The model sklearn.linear_model.Ridge implements ridge regression. You will use this model to examine a range of regularization strengths, comparing how they affect performance on the entire data-set.<sup>∗</sup>

<ul>

 <li>Using whichever polynomial degree you found to give the best performance on the data-set (after cross-validation, in the second part of this assignment), generate a series of Ridge models:

  <ul>

   <li>Each model should use a different <strong>regularization strength</strong>; this value, set using the model parameter alpha, measures the penalty applied to large weights (so that a higher value means that the model tries even harder to keep weights small). You will generate a series of 50 distinct weights (and so 50 different models), distributed logarithmically over the interval [0<em>.</em>01<em>,</em>100], using the code snippet:</li>

  </ul></li>

</ul>

np.logspace(-2, 2, base=10, num=50)

<ul>

 <li>You will do 5-fold cross-validation for each strength-value, again keeping track of average training and test error. (Do this the same way as in the prior question, using the supplied function to generate the folds required.)</li>

 <li>You will again produce a plot comparing these results. The plot should satisfy the guidelines of the one produced for the second part of the assignment, clearly showing the error results for each of the training and testing data. Again, a tabular version of the data should also be printed out.</li>

</ul>

<strong>Note</strong>: Since we are using a logarithmic scale for the strength values, the graphical plot should also use such a scale, to be most readable. This can be achieved using:

matplotlib.pyplot.xscale(‘log’)

<ul>

 <li>Discuss the plotted results. Where is the effect of increasing regularization strength helpful in avoiding overfitting, does it appear, and why do you say that? Where is the effect less useful, and why do you say that?</li>

</ul>