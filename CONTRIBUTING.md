## Can I open an issue with a PR?
I'd prefer not since there isn't much more to add, other than
quality-of-life features. Feel free to fork and make it your own: 
I'm using a MIT license!
## What needs to be done?
In order of decreasing priority:

* The ability to just pass the whole training sequence into the RNN, i.e.
handle the passing of the most recent hidden states within the class.
* Maybe the ability to handle some arbitrary, parameterized distribution for
the likelihood? Not sure how to do this nicely without becoming less PyTorch-like.
Would be complicated depending on whether or not you're using the expected value.
