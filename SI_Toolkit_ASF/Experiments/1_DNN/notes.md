# Notes

Data from a single car running hangar12. 10 experiments, split into 80/10/10. Control noise of 0.5.


# Dense-10IN-128H1-128H2-8OUT

## 0

Trained for 10 epochs.

Cannot always complete one lap, often drives into the wall in the chicane. Also, the car during data generation did this, maybe this is the issue here?

But it works for the beginning of the round, so that is good.

## 1

Trained for 30 epochs. Does not work as well as #0.

# Dense-10IN-128H1-128H2-128H3-8OUT

## 0

Worse than the smaller network, maybe overfits to dataset? Trained for 10 epochs. But validation loss does not go up again, maybe train for more epochs?

## 1

Trained for 40 epochs. Does not work well.

# Dense-10IN-64H1-64H2-8OUT

## 0

Works very well, also does not get hung up in chicane. Trained for 15 epochs.

## 1

Trained for 30 epochs. No big difference to 0, works very well.

# Dense-10IN-32H1-32H2-8OUT

## 0

Trained for 15 epochs. Completes round, but is more jittery than with 64/64 nodes.


# Dense-10IN-32H1-32H2-32H3-32H4-8OUT

## 0

Drives into wall, trained for 15 epochs.

# Open questions

- What happens if I train it for more epochs? Does the validation loss go up again? --> Not it does not, it stays constant at a certain point.

- How many layers are best?