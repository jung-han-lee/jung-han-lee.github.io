---
title: "Machine Learning with Julia: Handwritten digit recognition"
date: 2020-01-18
tags: [machine learning, data science, neural network, activation function, loss function]

excerpt: "Machine Learning, Neural Network, Data Science"
mathjax: "true"
---

# Handwritten digit recognition

## 1. Loading the data for the digits

We are now ready to train our neural network to tell apart two digits. For example, if we train a network to distinguish between a "1" and a "9", but then we give it a "0" instead, what should it do?

For now we will define the function **load_digit_data**, and call that function to load the data for the digits "1" and "9".

```julia
function load_digit_data(
        digits::Vector,
        nx::Integer=28,
        ny::Integer=28,
        nrep::Integer=1000
    )

    file0 = "data" * string(digits[1])
    file1 = "data" * string(digits[2])

    x0 = open(file0, "r") do file
        reshape(read(file), (nx, ny, nrep))
    end

    x1 = open(file1, "r") do file
        reshape(read(file), (nx, ny, nrep))
    end

    return x0, x1
end
```

and call that function to load the data for the digits "1" and "9".

```julia
digits = [1, 9]
nx, ny, nrep = 28, 28, 1000
x0, x1 = load_digit_data(digits)
```

## 2. Generating training & test datasets

We will now partition the dataset into a training and test dataset. We will learn to classify the digits on the training dataset and see how successful we are on the test dataset. The function following takes an input dataset and randomly partitions it into training and test datasets.

```julia
function generate_test_train_set_from_datacube(
        data::Array,
        percentage_train::Number=50.0
    )

    num_samples = size(data, 3)
    num_train = Integer(round(percentage_train / 100.0 * num_samples))
    num_test = num_samples - num_train

    rand_idx = randperm(num_samples)
    train_idx = rand_idx[1:num_train]
    test_idx = rand_idx[(num_train + 1):end]

    train_data = data[:, :, train_idx]
    test_data = data[:, :, test_idx]

    return train_data, test_data
end
```

The following code generates the testing and training data cubes.

```julia
x0_train, x0_test = generate_test_train_set_from_datacube(x0)
x1_train, x1_test = generate_test_train_set_from_datacube(x1);
```

To apply our method, we will have to convert the datacubes into matrices and ensure that the matrices have type accomplish this using the function defined.

```julia
function datacube2matrix(data_cube)
    return reshape(data_cube, :, size(data_cube, 3))
end
```

The next code converts the training and test data cube to matrices.

```julia
x0_train_matrix = datacube2matrix(x0_train)
x1_train_matrix = datacube2matrix(x1_train)
x0_test_matrix = datacube2matrix(x0_test)
x1_test_matrix = datacube2matrix(x1_test);
```


## 3. Encoding digits in a class Vector

```julia
function encode_class0_class1_data(
        class0_matrix::Matrix,
        class1_matrix::Matrix,
        encoding_vector::Vector=[0, 1]
    )
    num_class0 = size(class0_matrix, 2)
    num_class1 = size(class1_matrix, 2)
    class_matrix = hcat(class0_matrix, class1_matrix)
    class_vector = vcat(encoding_vector[1] * ones(num_class0), encoding_vector[2] * ones(num_class1))
    return class_matrix, class_vector
end
```

```julia
train_matrix, train_vector = encode_class0_class1_data(x0_train_matrix, x1_train_matrix, [-1, 1])
test_matrix, test_vector = encode_class0_class1_data(x0_test_matrix, x1_test_matrix, [-1, 1]);
```

## 4. Training neural network to classify digits with a **tanh** activation function

Now we train the network with the function learn2classify_sgd we defined in the earlier post.

```julia
mu = 1e-9
f_a = tanh
df_a = dtanh
@time w_hat, b_hat, loss = learn2classify_sgd(
    f_a, df_a, grad_loss,
    train_matrix, train_vector, mu,
    5000, 20, true, true, false
    );
```

and see how much lower the loss gets if we train our algorithm using the accelerated stochastic gradient algorithm.

```julia
@time w_hat, b_hat, loss = learn2classify_asgd(
    f_a, df_a, grad_loss,
    train_matrix, train_vector, mu,
    5000, 20, true, true, false
    );
```

## 5. Recognizing Handwritten digit using the functions.

We can draw in the box.
```julia
app = Canvas(
    brushsize=25
)
```

<img src="{{ site.url }}{{ site.baseurl }}/images/ml1/handwritten.png" alt="">

```julia
my_img_from_app = Gray.(image(app, (28, 28)))
my_img_from_app = float(1 .- my_img_from_app)
my_img = 255 * (my_img_from_app');
```
We can now use the neural network we trained to identify the transformed image corresponding to what we have written as in the next cell.

```julia
my_img_vector = (my_img[:])
class_prediction = sign.(gn(my_img[:], w_hat, b_hat, f_a))
my_digit_prediction = class2digit[class_prediction[]]
heatmap_digit(my_img; title="Network predicts $my_digit_prediction")
```

<img src="{{ site.url }}{{ site.baseurl }}/images/ml1/hand_result.png" alt="">


> Network predicts 1. Therefore, our model successfully recognize it!
