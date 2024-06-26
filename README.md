# handwriter

This repo contains implementation of deep recurrent network trained to generate human-like handwriting. 

The idea is heavily based on [1], but some things were simplified:

1. Network predicted exact (x, y) offset coordinates of next point and probability of stroke, thus having the output dimension of 3.

2. L2 loss for coordinates + entropy loss were used instead of mixed density loss because of greater numerical stability and faster calculation.

3. Dropout was used as regularization. No weight noise was applied, although it could be crucial for better inference performance, as suggested in [2].

4. Network was trained only on subset of available data and had 2(instead of 3) LSTM layers.

The model was trained on online handwriting dataset [3] using a subset of randomly drawn 8k samples as train dataset. The training was conducted using truncated backpropagation(named "tensorflow style" in [4]) in order to save memory. Random noise was added to training data to make network more stable at inference stage. Everything was implemented in python tensorflow [5].

Final model had around 3M parameters. Considering its limitation in comparison with model from [1], it still has provided some interesting results. The trained model demonstrated ability to generate readable letters, sometimes even placing them in sensible order. Below are 6 sequences of length 200, generated after model had seen sequences from training data of length 50(not shown).

![samples](./photos/samples.png)

All letters are divided clearly most probably because of end of stroke penalty set too high.

The trained model is stored under `model/model.zip`.

References:

[1] Graves A., "Generating Sequences With Recurrent Neural Networks", [https://arxiv.org/pdf/1308.0850.pdf](https://arxiv.org/pdf/1308.0850.pdf)

[2] Hinton G. et al., "Neural Networks for Machine Learning"(Lecture 9a), [https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec9.pdf](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec9.pdf)

[3] Liwicki M. and Bunke H., "IAM-onDB - an on-line English sentence database acquired from handwritten text on a whiteboard", [https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database)

[4] r2rt, "Styles of Truncated Backpropagation", [https://r2rt.com/styles-of-truncated-backpropagation.html](https://r2rt.com/styles-of-truncated-backpropagation.html)

[5] TensorFlow, [https://www.tensorflow.org](https://www.tensorflow.org)
