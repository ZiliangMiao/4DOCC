import torch
import numpy as np
import sys


# A CUSTOMIZED COLLATION FUNCTION TO ACCOMMODATE THE FACT THAT
# WE HAVE A DIFFERENT NUMBER OF POINTS FROM EVERY FRAME
# WE USE PADDING TO MAKE SURE THEY CAN BE BATCHED AS A 2D MATRIX
def CollateFn(batch):  # no map prior
    filenames = [item[0] for item in batch]

    # input points: N x max_{i} (sum_{t} n_{t, i}) x 3
    max_n_input_points = max([len(item[1]) for item in batch])
    input_points = torch.stack(
        [
            torch.nn.functional.pad(
                item[1],
                (0, 0, 0, max_n_input_points - len(item[1])),
                mode="constant",
                value=float("nan"),
            )
            for item in batch
        ]
    )  # pad with nan, make sure number of points is same in a batch

    # input tindex: N x max_{i} (sum_{t} n_{t, i})
    input_tindex = torch.stack(
        [
            torch.nn.functional.pad(
                item[2],
                (0, max_n_input_points - len(item[2])),
                mode="constant",
                value=-1,
            )
            for item in batch
        ]
    )

    # output origin: N x T x 3
    output_origin = torch.stack([item[3] for item in batch])

    # output points: N x max_{i} (sum_{t} n_{t, i}) x 3
    max_n_output_points = max([len(item[4]) for item in batch])

    output_points = torch.stack(
        [
            torch.nn.functional.pad(
                item[4],
                (0, 0, 0, max_n_output_points - len(item[4])),
                mode="constant",
                value=float("nan"),
            )
            for item in batch
        ]
    )

    # output tindex: N x max_{i} (sum_{t} n_{t, i})
    output_tindex = torch.stack(
        [
            torch.nn.functional.pad(
                item[5],
                (0, max_n_output_points - len(item[5])),
                mode="constant",
                value=-1,
            )
            for item in batch
        ]
    )

    if len(batch[0]) > 6:
        output_labels = torch.stack(
            [
                torch.nn.functional.pad(
                    item[6],
                    (0, max_n_output_points - len(item[6])),
                    mode="constant",
                    value=-1,
                )
                for item in batch
            ]
        )
        return (
            filenames,
            input_points,
            input_tindex,
            output_origin,
            output_points,
            output_tindex,
            output_labels
        )

    return (
            filenames,
            input_points,
            input_tindex,
            output_origin,
            output_points,
            output_tindex,
        )

def MinkCollateFn(batch):  # no map prior
    meta_info = [item[0] for item in batch]  # filenames

    # input 4d points: B * sum(N_i) x 4 (x, y, z, t)
    input_points_4d = torch.stack([item[1] for item in batch])

    # output origin: B x T_output x 3
    # T_input: {t_i}, T_output: {t_j}
    output_origin = torch.stack([item[2] for item in batch])

    # output points: B x sum(N_j) x 4
    output_points = torch.stack([item[3] for item in batch])

    return (meta_info,
            input_points_4d,
            output_origin,
            output_points)

