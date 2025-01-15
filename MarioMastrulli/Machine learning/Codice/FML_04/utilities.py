import matplotlib.pyplot as plt
import numpy as np


def plot_theta_gd(X, y, model, cost_history, theta_history, index_t0=0, index_t1=1):
    # Setup of meshgrid of theta values
    T0, T1 = np.linspace(theta_history[:, index_t0].min(), theta_history[:, index_t0].max(), 100), \
        np.linspace(theta_history[:, index_t1].min(), theta_history[:, index_t1].max(), 100)

    # Computing the cost function for each theta combination
    idx = np.random.randint(1000, size=100)
    zs = []
    for i, t0 in enumerate(T0):
        for q, t1 in enumerate(T1):
            model.theta[0] = t0
            model.theta[1] = t1
            h = X.dot(model.theta)
            j = (h - y)
            J = j.dot(j) / 2 / (len(X))
            zs.append(J)
    # Reshaping the cost values
    T0, T1 = np.meshgrid(T0, T1)
    Z = np.array(zs).reshape(T0.shape)

    anglesx = np.array(theta_history[:, index_t0])[1:] - np.array(theta_history[:, index_t0])[:-1]
    anglesy = np.array(theta_history[:, index_t1])[1:] - np.array(theta_history[:, index_t1])[:-1]

    fig = plt.figure(figsize=(16, 8))

    # Surface plot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(T0, T1, Z, rstride=5, cstride=5, cmap='jet', alpha=0.5)
    ax.plot(theta_history[:, index_t0], theta_history[:, index_t1], cost_history, marker='*',
            color='r', alpha=.4, label='Gradient descent')

    ax.set_xlabel('theta 0')
    ax.set_ylabel('theta 1')
    ax.set_zlabel('Cost function')
    ax.set_title('Gradient descent: Root at {}'.format(model.theta.ravel()))
    ax.view_init(45, 45)

    # Contour plot
    ax = fig.add_subplot(1, 2, 2)
    ax.contour(T0, T1, Z, 70, cmap='jet')
    ax.quiver(theta_history[:, index_t0][:-1], theta_history[:, index_t1][:-1], anglesx, anglesy,
              scale_units='xy', angles='xy', scale=1, color='r', alpha=.9)

    plt.show()