# DeepAR in pytorch
## What?
An implementation of the DeepAR forecasting framework in PyTorch for regression tasks [1].

As in the original paper, Gaussian log-likelihood and LSTMs are used. The code,
however, allows the user to input their own RNNs. A bit more wrangling
is needed to support non-Gaussian likelihood: just switch the Gaussian distribution
parameters with those of yours.
## Why?
Originally written when I was doing research at UMD, I wasn't a fan of how 
existing PyTorch implementations were bundled into separate packages that
differed from the usual way of using PyTorch. This code only extends `nn.Module`.
It's just that simple.
## How?
An iterative forecasting example is included since one would need to step through
the sequence individually.
For RNNs in PyTorch proper, users can just pass in the whole training sequence.
In other words, the RNN class handles the passing of the most recent hidden 
state to the most recent RNN cell at each time-step.

I didn't include this functionality due to time constraints. I'll maybe include
it in the future since I learned a lot of undocumented features looking at the `nn.Module` 
source code.

## References
[1] - David Salinas, et al. "DeepAR: Probabilistic forecasting with autoregressive recurrent networks". International Journal of Forecasting 36. 3(2020): 1181-1191. https://www.sciencedirect.com/science/article/pii/S0169207019301888
[2] - Lim Bryan and Zohren Stefan. 2021 Time-series forecasting with deep learning: a survey. Phil. Trans. R. Soc. A. 379: 20200209 20200209. http://doi.org/10.1098/rsta.2020.0209

# Training Suggestions
In my experience, RNNs generally sucked when it came to iterative forecasting.
After all, with each step, I'd imagine the model weights wouldn't be able to 
change as significantly as needed to maintain any stochastic process with 
autocorrelation.

I found greater success with direct forecasting, especially with min-max scaling.
Of course, one would have to _a priori_ know the range of forecasted values with scaling.
That said, I'm skeptical of distribution sampling for forecasting, especially if you
just use the expected value.

(Here, "direct forecasting" and "iterative forecasting" are defined as in [2])
