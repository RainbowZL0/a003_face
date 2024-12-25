import os

import torch

from a002_model.a001_utils.a002_general_utils import get_time_stamp_str
from a002_model.a003_training.a004_quant_model import generate_my_facenet_model, convert_model_to_int8


def get_quantization_model(state_path):
    model = generate_my_facenet_model(with_quantization=True, pretrained='vggface2', device='cpu')
    model = model.eval().to('cpu')
    model = convert_model_to_int8(model)

    read_state = torch.load(state_path, map_location='cpu')
    model.load_state_dict(read_state["model_state"], strict=False)

    return model


def start_analysis():
    model = get_quantization_model(
        r"a002_model/a003_training/saved_history/models"
        r"/2024-12-24_14-03-58_epochs-2_iters-up-to-now-312.pth"
    )
    test_data = torch.rand(
        size=(1, 3, 160, 160),
        dtype=torch.float32,
        device='cpu',
    )

    # 使用 torch.profiler 进行性能分析
    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                os.path.join(
                    'a002_model/a003_training/analysis',
                    get_time_stamp_str()
                )
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
    ) as prof:
        for _ in range(6):
            prof.step()
            with torch.no_grad():
                model(test_data)


if __name__ == '__main__':
    start_analysis()
