import torch
from torch.nn import functional
from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from torch import optim
# noinspection PyProtectedMember
from torch.quantization import (
    QuantStub,
    DeQuantStub,
    get_default_qat_qconfig,
    prepare_qat,
    convert,
)

from a002_model.a001_utils.a000_CONFIG import TRAINING_INITIAL_LR


class MyFacenetModel(InceptionResnetV1):
    def __init__(self, with_quantization, pretrained):
        self.with_quantization = with_quantization
        self.pretrained = pretrained

        super().__init__(pretrained=pretrained)

        if with_quantization:
            # 量化准备
            self.quant = QuantStub()
            self.dequant = DeQuantStub()

    def forward(self, x):
        if self.with_quantization:
            x = self.quant(x)

        x = super().forward(x)

        if self.with_quantization:
            x = self.dequant(x)

        x = self.last_bn(x)
        x = functional.normalize(x, p=2, dim=1)

        return x


def test_main():
    model = generate_my_facenet_model(with_quantization=True, pretrained='vggface2', device='cpu')

    ts = torch.randn(64, 3, 160, 160)
    rst = model(ts)

    optimizer = optim.Adam(model.parameters(), lr=TRAINING_INITIAL_LR)

    optimizer.zero_grad()

    loss = rst.mean()
    loss.backward()

    optimizer.step()

    convert_model_to_int8(model)
    pass


def generate_my_facenet_model(with_quantization, pretrained, device):
    model = MyFacenetModel(with_quantization=with_quantization, pretrained=pretrained).to(device)
    if with_quantization:
        model.qconfig = get_default_qat_qconfig()  # 'fbgemm' 是适用于 x86 CPU的过时写法，当前建议用'x86'，也是默认值
        model = prepare_qat(model.train())
    return model


def convert_model_to_int8(model):
    return convert(model)


def resave_state_dict():
    """
    在facenet-pytorch库中，有一些参与运算的常数没有被注册为buffer，导致量化操作时它们仍然是float类型，但其他数值是int，
    导致运算类型冲突。解决方法是修改源码，把它们改成register_buffer。然而，加载预训练的权重时，又会遇到key不匹配的问题，
    因为state文件中没有我们定义的buffer参数。解决方法是，先用strict=False读取预训练权重，覆盖buffer以外的其他参数，此时模型内
    state是完整的，包括buffer。然后我们保存这一套完整的state。
    facenet_pytorch/models/inception_resnet_v1.py
    """
    model = InceptionResnetV1(pretrained='vggface2')
    # state = torch.load('C:\\Users\\46733/.cache\\torch\\checkpoints\\20180402-114759-vggface2.pt')
    # missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    torch.save(model.state_dict(), "D:/Desktop/_Search/Desktop2/with_buffer.pt")


if __name__ == '__main__':
    # test_main()
    resave_state_dict()
