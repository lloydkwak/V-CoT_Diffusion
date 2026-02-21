import torch
import torch.nn as nn
import pytorch3d.transforms as pt

class RotationTransformer(nn.Module):
    def __init__(self, from_rep='axis_angle', to_rep='rotation_6d'):
        super().__init__()
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, x):
        if self.from_rep == self.to_rep:
            return x
        elif self.from_rep == 'axis_angle' and self.to_rep == 'rotation_6d':
            # axis_angle [..., 3] -> rotation_6d [..., 6]
            matrix = pt.axis_angle_to_matrix(x)
            return pt.matrix_to_rotation_6d(matrix)
        elif self.from_rep == 'quaternion' and self.to_rep == 'rotation_6d':
            # quaternion [..., 4] -> rotation_6d [..., 6]
            matrix = pt.quaternion_to_matrix(x)
            return pt.matrix_to_rotation_6d(matrix)
        else:
            raise NotImplementedError(f"Conversion from {self.from_rep} to {self.to_rep} is not implemented.")

    def inverse(self, x):
        if self.from_rep == self.to_rep:
            return x
        elif self.from_rep == 'axis_angle' and self.to_rep == 'rotation_6d':
            # rotation_6d [..., 6] -> axis_angle [..., 3]
            matrix = pt.rotation_6d_to_matrix(x)
            return pt.matrix_to_axis_angle(matrix)
        else:
            raise NotImplementedError(f"Inverse conversion from {self.to_rep} to {self.from_rep} is not implemented.")
