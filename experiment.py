import numpy as np
import random
import matplotlib.pyplot as plt
import sys

from myrobot import myRRrobot, RR_model1

def RR_experiment(dp, disturbances, train=False, saveafter=False):
    myRR = RR_model1(50,50,'right',\
        './models/invRR_right/invkinRR_right.ckpt',\
        './models/compRR_right/internalforwardRR_right.ckpt',\
        './models/compRR_right/internalinverseRR_right.ckpt')

    myRR.setPID(0.5, 0.05, 1)

    myRR.init_inverseNN()
    myRR.open_inverseNN_session()
    if train:
        myRR.train_inverseNN(display_step=1000)
        myRR.save_inverseNN()
    #myRR.test_IF_NN(10)

    myRR.init_IF_NN()
    myRR.open_IF_NN_session()
    myRR.init_II_NN()
    myRR.open_II_NN_session()
    if train:
        myRR.train_IF_NN(display_step=1000)
        myRR.save_IF_NN()
        myRR.train_II_NN(display_step=1000)
        myRR.save_II_NN()


    x = np.array(dp)
    noloop = myRR.forward(myRR.inverseSingle(x))[0:2,3]
    withloop = myRR.reach(x, learn=False)
    print(noloop)
    print(withloop)
    #sys.exit()
# experiment setup
    displacements = []
    for d in disturbances:
        for i in range(d[1]):
            displacements.append(d[0])
    total_iters = sum([d[1] for d in disturbances])

    errors = []
    x_actuals = []
# learn
    for i in range(total_iters):
        x_actual = myRR.reach(x, learn=True, learning_rate=0.01, num_steps=1,\
            Y_disturbance=np.array(displacements[i]))
        x_actuals.append(x_actual)
        error = np.linalg.norm(x_actual-x)
        errors.append(error)

# close session
    if saveafter:
        myRR.save_inverseNN()
        myRR.save_IF_NN()
        myRR.save_II_NN()
    myRR.close_II_NN_session()
    myRR.close_IF_NN_session()
    myRR.close_inverseNN_session()

# graph
    x_axis = np.array([i for i in range(total_iters)])[20:]
    norm_errors = np.array(errors)[20:]
    x_coords = np.array(x_actuals)[20:,0]
    y_coords = np.array(x_actuals)[20:,1]

    plt.figure(1)

    plt.subplot(311)
    plt.plot(x_axis, norm_errors)
    plt.grid(True)

    plt.subplot(312)
    plt.plot(x_axis, x_coords)
    plt.grid(True)

    plt.subplot(313)
    plt.plot(x_axis, y_coords)
    plt.grid(True)

    #y_axis = np.array(x_actuals)
    plt.show()

if __name__ == "__main__":
    desired_point = [30,60]
    disturbances = [
        [[0,0],150],
        [[10,0],150],
        [[0,0],150],
        [[10,0],150],
        [[0,0],150],
        [[10,0],150],
    ]
    RR_experiment(desired_point, disturbances, train=False)