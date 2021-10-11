import statistics
import numpy as np
import matplotlib.pyplot as plt
import seaborn

seaborn.set(style='ticks')


def function(x: float) -> float:
    return x**5-x**3+3*(x**2)-x+10


def generate_x_y(left_border: float, right_border: float, number: int) -> np.ndarray(tuple):

    x = np.linspace(left_border, right_border, number)
    y = function(x)
    return np.vstack((x, y)).T


def get_system_of_equations(x, y, n):
    xs = np.array([])
    xy = np.array([])
    for index in range(0, (n + 1)):
        for exp in range(0, (n + 1)):
            tx = np.sum(x**(index + exp))
            xs = np.append(xs, tx)
        ty = np.sum(y * (x**index))
        xy = np.append(xy, ty)
    return xs, xy


def polynomial_fit(xy_pairs, k):
    xs, xy = get_system_of_equations(xy_pairs[:, 0], xy_pairs[:, 1], k)
    xs = np.reshape(xs, ((k + 1), (k + 1)))
    xy = np.reshape(xy, ((k + 1), 1))
    a = np.linalg.solve(xs, xy).flatten()
    return a


def func(x, arguments):
    y = 0
    for i in range(len(arguments)):
        y += arguments[i]*(x**i)
    return y


def losses(real_y, y):
    tmp_arr = []
    for i in range(len(y)):
        tmp_arr.append(pow(real_y[i]-y[i], 2))
    return sum(tmp_arr)


def process_power(dict_to_insert, arguments, data_set, power):
    x = data_set[:, 0]
    y = func(x, arguments)
    if power not in dict_to_insert:
        dict_to_insert[power] = [losses(data_set[:, 1], y)]
    else:
        dict_to_insert[power].append(losses(data_set[:, 1], y))


def process_mean(dict_to_insert, dict_to_process):
    for power, loss in dict_to_process.items():
        if power not in dict_to_insert:
            dict_to_insert[power] = [statistics.mean(loss)]
        else:
            dict_to_insert[power].append(statistics.mean(loss))


def main():
    data_splited = np.split(generate_x_y(-49, 50, 100), 5)
    training_losses = {}
    validation_losses = {}

    for validation_data in range(len(data_splited)):
        losses_train = {}
        losses_val = {}
        for data_set in range(len(data_splited)):
            if validation_data == data_set:
                continue
            else:
                for power in range(0, 11):
                    arguments = polynomial_fit(data_splited[data_set], power)

                    process_power(losses_train, arguments,
                                  data_splited[data_set], power)

                    process_power(losses_val, arguments,
                                  data_splited[validation_data], power)

        process_mean(training_losses, losses_train)
        process_mean(validation_losses, losses_val)

    train = []
    validate = []
    for power, losses_in_train in training_losses.items():
        train.append([power, statistics.mean(losses_in_train)])
    for power, losses_in_validation in validation_losses.items():
        validate.append([power, statistics.mean(losses_in_validation)])

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Mean error depending on the degree of the polynomial")
    ax1.set_title("Training error")
    ax2.set_title("Validation error")
    ax1.grid(True, which='both')
    ax2.grid(True, which='both')
    seaborn.despine(ax=ax1, offset=0)
    seaborn.despine(ax=ax2, offset=0)

    ax1.plot([row[0] for row in train], [row[1] for row in train])
    ax2.plot([row[0] for row in validate], [row[1] for row in validate])

    plt.show()


if __name__ == '__main__':
    main()
