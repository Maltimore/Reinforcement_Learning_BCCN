import matplotlib.pyplot as plt
import numpy as np



def is_between(a, b, c):
    a[0], a[1] = round(a[0], 3), round(a[1], 3)
    b[0], b[1] = round(b[0], 3), round(b[1], 3)
    c[0], c[1] = round(c[0], 3), round(c[1], 3)
    print(a)
    print(b)
    print(c)
    print((np.isclose((b[0] - a[0]) * (c[1] - a[1]), (c[0] - a[0]) * (b[1] - a[1]))), \
          (((a[0] <= c[0]) and (b[0] >= c[0])) or ((a[0] >= c[0]) and (b[0] <= c[0]))), \
          (((a[1] <= c[1]) and (b[1] >= c[1])) or ((a[1] >= c[1]) and (b[1] <= c[1]))))
    return (np.isclose((b[0] - a[0]) * (c[1] - a[1]), (c[0] - a[0]) * (b[1] - a[1])) and
        (((a[0] <= c[0]) and (b[0] >= c[0])) or ((a[0] >= c[0]) and (b[0] <= c[0]))) and
        (((a[1] <= c[1]) and (b[1] >= c[1])) or ((a[1] >= c[1]) and (b[1] <= c[1]))))






# initialize values
q0 = [ 59.34824,   0.23125]
q1 = [62.657629999999997, 0.23124]


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
    print("index is: " + str(i))
    if is_between(startpoints[i,:], endpoints[i,:], px) and is_between(q0, q1, px):
        print("The line sected is number " + str(i))
        plt.scatter(px[0], px[1])
plt.plot([q0[0], q1[0]],[q0[1],q1[1]])
plt.xlim([-10, 150])
plt.ylim([-10, 70])
plt.legend()

vector_a = np.array([7,7], dtype=float)
vector_b = np.array([.5,.5], dtype=float)
#vector_a /= np.linalg.norm(vector_a)
#vector_b /= np.linalg.norm(vector_b)
print(np.dot(vector_a, vector_b))