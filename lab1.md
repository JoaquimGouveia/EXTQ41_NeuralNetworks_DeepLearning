# The report!

We have added intructions inside this report template. As you write your report, remove the instructions.

## Name : **Joaquim Gouveia**

## Introduction
A few sentences about the overall theme of the exercise.

## Answers to questions
Provide enough information to clarify the meaning of your answers, so that they can be understood by someone who does not scroll up and read the entire instruction.

The questions are repeated here, for clarity of what is demanded. If it does not fit your style to quote them verbatim, change the format. 

#### Question 1, variations in pre-defined MLP
(a) Do you see the same loss vs epoch behavior each time your run? If not, why?
**Answer:** No, the loss vs epoch behaviour is not the same everytime I run the code. This might be due to different factors, for example: the main reason is probably due to the random initialization of the weights in the network. Basically, different initial weights can lead the model to explore different parts of the loss surface, which may converge to different local minima. This leads to the next possible explanation, the tanh activation function introduces nonlinearity, making the loss function non-convex. This means that the loss function has multiple local minima, and the optimization algorithm may converge to different local minima depending on the initial weights. The fixed learning rate of 0.05 (between epochs) might also work better for some initializations than others and may cause the model to converge differently depending on the quality of the initial weights, with some runs oscillating near suboptimal solutions. Furthermore the model only contains 4 hidden nodes in 1 layer, which also makes it even more sensitive to initialization and less capable to consistently learn the data.

(b) Do you observe that training fails, i.e. do not reach low loss, during any of these five runs?
**Answer:** Loss in first run: 0.4425813556
            Loss in second run: 0.4679292440
            Loss in third run: 0.0457922257
            Loss in fourth run: 0.0023997910
            Loss in fifth run: 0.0017875287
            The training fails in several of the runs, as the loss does not reach < 0.001.

#### Question 2, vary learning rate
Present your average MSE results and discuss your findings.
**Answer:** Average MSE Learning rate 0.5: NaN (too high)
            Average MSE Learning rate 0.2: 0.0011907210666667
            Average MSE Learning rate 0.1: 0.0039310288
            Average MSE Learning rate 0.05 (3 new runs): 0.0030344165666667
            Average MSE Learning rate 0.02: 0.2491593789333333
            **Discussion of findings:** As shortly mentioned in the previous question, the learning rate is a very important hyperparameter to tune. The results show how the learning rate influences the model's ability to train and minimize the loss. A learning rate that is too high (ex: 0.5) results in unstable training, with the loss diverging instead of decreasing. The model cannot converge to a meaningful minimum, which is why the MSE becomes undefined or too high too calculate. A learning rate that is too low (ex: 0.02) results in slow training, as the model takes too small steps in the loss surface. This makes the model converge very slowly, and in some cases, it might not even reach a good solution. Furthermore too low learning rates can also make the model get stuck in local minima or saddle points. The best learning rate seems to be around 0.1-0.2, as the model converges to a good solution in a reasonable amount of time as well as being able to leave local minima.

#### Question 3, vary (mini)batch size
Present and discuss your findings.
**Answer:** Training with batch size of 50: **Successful?:** Yes (MSE: 0.0023689858) **Number of necessary Epochs:** Approximately 3000
            Training with batch size of 25: **Successful?:** Yes (MSE: 0.0015719042) **Number of necessary Epochs:** Approximately 1250
            Training with batch size of 10: **Successful?:** Yes (MSE: 0.0011349532) **Number of necessary Epochs:** Approximately 750
            **Discussion of findings:** The batch size is another important hyperparameter to tune. The results demonstrate how the batch size affects the training efficiency, convergence speed, and performance of the model when using stochastic gradient descent (SGD). The batch size of 50, as previously discussed, uses the entire dataset for each graident update, making this equivalent to ordinary gradient descent (GD). This method is slower to converge since each weight update only occurs after the entire dataset has been processed. The batch size of 10, on the other hand, uses only 10 samples for each weight update, making the model converge faster (750 epochs). Additionally, the smaller batch size introduces more randomness (increases variance of weight updates, leading to noisier convergence behaviour), which can help the optimizer escape local minima and improve final performance. In this experience, a batch size of 10 seems to be optimal, achieving the best performance (lowest MSE) and fastest convergence (around 750 epochs).

#### Question 4, select good hyper-parameters
Present your best combination of learning rate and batch size, and its result.
**Answer:** After the past two questions it seemed as if the best combination would be a learning rate of 0.2 and a batch size of 10. This is not correct though since the loss vs epoch behaviour became erratic and very volatile. A batch size of 10 introduces randomness into the gradient updates because the gradient is calculated from only a small subset of the training data each time. If we set the learning rate to 0.2, which can be considered very high for this batch size, the magnitude of each weight update will be amplified quite a lot, leading to instability. After several combinations later I have reached the conclusion that one possible optimal combination is a learning rate of 0.2 and batch size of 25 (possibly also learning rate of 0.11 and batch size of 13).

#### Question 5, vary epochs
Compare the number of epochs needed to reach a good solution with that of Q4. If you cannot find a good solution in a reasonable number of epochs, you can "revert" the problem: optimize learning rate and batch size for Q5, and the see how those hyper-parameters perform on Q4.

#### Question 6, vary network size and other hyper-parameters
Present your set of good hyper-parameters and the result. 

Note: If you cannot solve this task in *reasonable* time, present your best attempt!

#### Question 7, optimize hyper-parameters for classification
Present your set of hyper-parameters that reach > 95% accuracy

#### Question 8, change learning algorithm
Try the Adam optimizer for Q7, and compare (qualitatively) the results and the number of epochs needed to get them. Present changes you needed to make to improve the results of the Adam optimizer, if any.

#### Bonus tasks (if you feel inspired)

## Summary
Connect the summary to your introduction, to provide a brief overview of your findings.
  