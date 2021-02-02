# Results

In general, non-normalized data explodes some calculation.
In sigmoid, it explodes the exponential wX+b; in linear, because of A^2 in the cost.

In sigmoid example, the learning rate can be really big like 100 and the code optimizes well (but it's an easy dataset). With smaller learning rates, we need more cycles.

## Interesting

The backward propagation ends up being the same for both methods (apart from a constant as far as I can see). So it is useful for both methods. The cost is different though.
