import torch
import torch.nn as nn

"""
    This function is implemented IoU as a NN block.
    Input: 2 bounding boxes, B_g = (x_g, y_g, w_g, h_g) and B_d = (x_d, y_d, w_d, h_d)
    
    Intersection Calculation:
        x_{i1} &= \max(x_g, x_d),
        y_{i1} &= \max(y_g, y_d),
        x_{i2} &= \min(x_g + w_g, x_d + w_d),
        y_{i2} &= \min(y_g + h_g, y_d + h_d).

    The width and height of the intersection are given by:
        w_{i} &= \max(0, x_{i2} - x_{i1}),
        h_{i} &= \max(0, y_{i2} - y_{i1}).
    
    The area of the intersection A_{I} is w_{i} \cdot h_{i}.

    Union Calculation:
    The areas of each bounding box are:
        A_g &= w_g \cdot h_g,
        A_d &= w_d \cdot h_d.
    The area of the union is:
        A_{U} = A_g + A_d - A_{I}.
        
    IoU Calculation and Constraint:
    The IoU is defined as:
        IoU = \frac{A_{I}}{A_{U}}.
        
    To enforce the constraint \( IoU > \tau \), we express this as a linear constraint,
    which can be encoded using ReLU activation:
        A_{I} > \tau \cdot A_{U} \Leftrightarrow  A_{I} - \tau \cdot A_{U} > 0.
        
    Each part will be encoded as a separate layer in the NN, using weights and biases.
"""


class MaxLayer(nn.Module):
    def __init__(self):
        super(MaxLayer, self).__init__()

    def forward(self, x1, x2):
        return x1 + torch.relu(x2 - x1)


class MinLayer(nn.Module):
    def __init__(self):
        super(MinLayer, self).__init__()

    def forward(self, x1, x2):
        return -MaxLayer()(-x1, -x2)


class IoU(nn.Module):
    def __init__(self, tau=0.5):
        super(IoU, self).__init__()
        self.tau = tau
        # Intersection Calculation
        self.linear1 = nn.Linear(8, 8, bias=False)
        # init weights and biases. The order is: x_g, y_g, w_g, h_g, x_d, y_d, w_d, h_d
        # The output of the layer should be:
        # x_g, y_g, x_g + w_g, y_g + h_g, x_d, y_d, x_d + w_d, y_d + h_d
        self.linear1.weight.data = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 0, 0, 0, 0, 0],
                                                    [1, 0, 1, 0, 0, 0, 0, 0],
                                                    [0, 1, 0, 1, 0, 0, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 0, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 0],
                                                    [0, 0, 0, 0, 1, 0, 1, 0],
                                                    [0, 0, 0, 0, 0, 1, 0, 1]], dtype=torch.float32)

        self.linear2 = nn.Linear(4, 2, bias=False)
        # init weights and biases. The order is: x_i1, y_i1, x_i2, y_i2
        # x_i2 - x_i1, y_i2 - y_i1
        self.linear2.weight.data = torch.tensor([[-1, 0, 1, 0],
                                                 [0, -1, 0, 1]], dtype=torch.float32)

        # w_i * h_i will be calculated in the forward method

        # Union Calculation
        # w_g * h_g, w_d * h_d will be calculated in the forward method

        # A_{U} = A_g + A_d - A_{I}
        self.linear3 = nn.Linear(3, 1, bias=False)
        # init weights and biases. The order is: A_g, A_d, A_{I}
        # A_g + A_d - A_{I}
        self.linear3.weight.data = torch.tensor([[1, 1, -1]], dtype=torch.float32)

        # IoU Calculation and Constraint
        self.linear4 = nn.Linear(2, 1, bias=False)
        # init weights and biases. The order is: A_{I}, A_{U}
        # A_{I} - tau * A_{U}
        self.linear4.weight.data = torch.tensor([[1, -self.tau]], dtype=torch.float32)


    def forward(self, Input):
        # first layer: intersection calculation
        Inter = self.linear1(Input)
        # max between x_g and x_d, y_g and y_d, min between x_g + w_g and x_d + w_d, y_g + h_g and y_d + h_d
        Inter_max_min = torch.stack([
            MaxLayer()(Inter[:, 0], Inter[:, 4]),
            MaxLayer()(Inter[:, 1], Inter[:, 5]),
            MinLayer()(Inter[:, 2], Inter[:, 6]),
            MinLayer()(Inter[:, 3], Inter[:, 7])
        ], dim=1)

        # print(f"x_i1 = {Inter_max_min[:, 0]}, y_i1 = {Inter_max_min[:, 1]}, x_i2 = {Inter_max_min[:, 2]}, y_i2 = {Inter_max_min[:, 3]}")

        # second layer: width and height calculation
        Inter = self.linear2(Inter_max_min)

        # max between x_i2 - x_i1 and 0, y_i2 - y_i1 and 0, use MaxLayer to implement ReLU
        Inter = torch.relu(Inter)
        # Area = w_i * h_i
        Area = Inter[:, 0] * Inter[:, 1]

        # print(f"w_i = {Inter[:, 0]}, h_i = {Inter[:, 1]}")

        # Union Calculation
        # A_{U} = A_g + A_d - A_{I}
        # since Area is zero-dimensional tensor, we need to expand it to 1-dimensional tensor,
        # or reduce the dimension of the other tensors
        Input_area_g = (Input[:, 2] * Input[:, 3]).unsqueeze(1)
        Input_area_d = (Input[:, 6] * Input[:, 7]).unsqueeze(1)
        Area_expanded = Area.unsqueeze(1)
        Union = self.linear3(torch.cat([Input_area_g, Input_area_d, Area_expanded], dim=1))
        # print(f"IoU using our encoding:\nInter = {Area}, Union = {Union} \nIoU = {Area / Union.squeeze(1)}\n")
        # print(f"IoU = {Area / Union.squeeze(1)}\n")
        # IoU Calculation and Constraint
        z = self.linear4(torch.cat([Area.unsqueeze(1), Union], dim=1))

        return z

# common IoU
def _IoU(B_g, B_d):
    x_i1 = torch.max(B_g[:, 0], B_d[:, 0])
    y_i1 = torch.max(B_g[:, 1], B_d[:, 1])
    x_i2 = torch.min(B_g[:, 0] + B_g[:, 2], B_d[:, 0] + B_d[:, 2])
    y_i2 = torch.min(B_g[:, 1] + B_g[:, 3], B_d[:, 1] + B_d[:, 3])

    w_i = torch.max(torch.zeros_like(x_i1), x_i2 - x_i1)
    h_i = torch.max(torch.zeros_like(y_i1), y_i2 - y_i1)

    # print(f"x_i1 = {x_i1}, y_i1 = {y_i1}, x_i2 = {x_i2}, y_i2 = {y_i2}")
    # print(f"w_i = {w_i}, h_i = {h_i}")

    A_I = w_i * h_i

    A_g = B_g[:, 2] * B_g[:, 3]
    A_d = B_d[:, 2] * B_d[:, 3]

    A_U = A_g + A_d - A_I

    print(f"IoU using traditional method:\nInter = {A_I}, Union = {A_U} \nIoU = {A_I / A_U}\n")

    return A_I / A_U


def test_IoU():
    B_g = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
    B_d = torch.tensor([[5, 5, 10, 10]], dtype=torch.float32)


    z = _IoU(B_g, B_d)

    iou = IoU(tau=0.5)
    z = iou(torch.cat([B_g, B_d], dim=1))

    # test2, different x,y coordinates
    print("\nTest 2\n")
    B_g = torch.tensor([[0, 1, 5, 10]], dtype=torch.float32)
    B_d = torch.tensor([[4, 3, 4, 9]], dtype=torch.float32)

    z = _IoU(B_g, B_d)

    iou = IoU(tau=0.5)
    z = iou(torch.cat([B_g, B_d], dim=1))


    # Now run both examples together as a batch
    B_g = torch.tensor([[0, 0, 10, 10], [0, 1, 5, 10]], dtype=torch.float32)
    B_d = torch.tensor([[5, 5, 10, 10], [4, 3, 4, 9]], dtype=torch.float32)
    z = iou(torch.cat([B_g, B_d], dim=1))



# main
if __name__ == "__main__":
    test_IoU()