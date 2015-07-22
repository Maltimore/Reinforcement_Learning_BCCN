import matplotlib.pyplot as plt
import numpy as np



def is_between(a, b, c):
    return (np.isclose((b[0] - a[0]) * (c[1] - a[1]), (c[0] - a[0]) * (b[1] - a[1])) and
        (((a[0] <= c[0]) and (b[0] >= c[0])) or ((a[0] >= c[0]) and (b[0] <= c[0]))) and
        (((a[1] <= c[1]) and (b[1] >= c[1])) or ((a[1] >= c[1]) and (b[1] <= c[1]))))






# initialize values
q0 = [50.133629557470982, 3.1240315663402529]
q1 = [45.632582188115137, 3.1240315663402534]


def intersection(q0, q1, p0, p1):
    dy = q0[1] - p0[1]
    dx = q0[0] - p0[0]
    lhs0 = [-dy, dx]
    rhs0 = p0[1] * dx - dy * p0[0]
    
    dy = q1[1] - p1[1]
    dx = q1[0] - p1[0]
    lhs1 = [-dy, dx]
    rhs1 = p1[1] * dx - dy * p1[0]
    
    a = np.array([lhs0, 
                  lhs1])
    
    b = np.array([rhs0, 
                  rhs1])
    try:
        px = np.linalg.solve(a, b)
    except:
        px = np.array([np.nan, np.nan])
    return px
    
startpoints = np.array([[0, 60],
                        [0, 50],
                        [60, 50],
                        [50, 0],
                        [0, 50],
                        [50, 0],
                        [60, 0],
                        [110, 50]])
endpoints = np.array([[110, 60],
                      [50, 50],
                      [110, 50],
                      [60, 0],
                      [0, 60],
                      [50, 50],
                      [60, 50],
                      [110, 60]])
for i in np.arange(8):
    plt.plot([startpoints[i,0], endpoints[i,0]],[startpoints[i,1], endpoints[i,1]], label="nr " + str(i))

    px = intersection(startpoints[i,:], q0, endpoints[i,:], q1)
    if is_between(startpoints[i,:], endpoints[i,:], px) and is_between(q0, q1, px):
        print("The line sected is number " + str(i))
        plt.scatter(px[0], px[1])
plt.plot([q0[0], q1[0]],[q0[1],q1[1]])
plt.xlim([-10, 150])
plt.ylim([-10, 70])
plt.legend()