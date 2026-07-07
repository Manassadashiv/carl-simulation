import numpy as np
MAP_RES = 25
CM = np.zeros((MAP_RES, MAP_RES, 5))
def diffuse_scent(CM, goal_pos):
    gi, gj = 18, 18
    scent = CM[:, :, 3]
    danger = CM[:, :, 0]
    ns = np.zeros_like(scent)
    ns[1:-1, 1:-1] = (scent[:-2, 1:-1] + scent[2:, 1:-1] + scent[1:-1, :-2] + scent[1:-1, 2:])
    new_scent = scent.copy()
    new_scent[1:-1, 1:-1] += 0.25 * (ns[1:-1, 1:-1] - 4*scent[1:-1, 1:-1])
    new_scent[danger > 0.5] = 0.0
    new_scent *= 0.999
    new_scent[gi, gj] = 1.0
    CM[:, :, 3] = new_scent
    return CM
for _ in range(10000):
    CM = diffuse_scent(CM, (5.4, 5.4))
print("Scent at 2,2:", CM[2, 2, 3])
