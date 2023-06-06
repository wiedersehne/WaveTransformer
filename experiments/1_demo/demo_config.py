import numpy as np

def get_demo_config():

    config = {
            "bias": [[-0.5, 0.5],
                     [0.5, -0.5],
                     [-0.5, 0.5],
                     [-0.5, 0.5],
                     [0, 0],
                     [0, 0]],
            "base_angular_freq": [[1 * np.pi, 2 * np.pi],
                                  [1 * np.pi, 2 * np.pi],
                                  [1 * np.pi, 1 * np.pi],
                                  [1 * np.pi, 1 * np.pi],
                                  [3 * np.pi, 2 * np.pi],
                                  [3 * np.pi, 2 * np.pi]],
            "base_amplitude": [[0.5, 0.5],
                               [0.5, 0.5],
                               [0.0, 0.5],
                               [0.0, 0.5],
                               [0.5, 0.5],
                               [0.5, 0.5]],
            "transient_bool": [False, False, False, True, False, False],
            "transient_start": [[np.NaN, np.NaN],
                                [np.NaN, np.NaN],
                                [np.NaN, np.NaN],
                                [125, 250],
                                [np.NaN, np.NaN],
                                [np.NaN, np.NaN]],
            "transient_amplitude": [[np.NaN, np.NaN],
                                    [np.NaN, np.NaN],
                                    [np.NaN, np.NaN],
                                    [0.15, -0.15],
                                    [np.NaN, np.NaN],
                                    [np.NaN, np.NaN]],
            "singularity_bool": [False, False, False, False, False, True],
            "singularity_start": [[np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [np.NaN, np.NaN],
                                  [50, 300]],
            "singularity_amplitude": [[np.NaN, np.NaN],
                                      [np.NaN, np.NaN],
                                      [np.NaN, np.NaN],
                                      [np.NaN, np.NaN],
                                      [np.NaN, np.NaN],
                                      [-0.5, 0.5]],
        }
    return config