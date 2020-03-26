import os
import json
import time

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from mlsploit_local import Job

from data import (
    build_output_figure_dataset,
    fig2arr,
    get_or_create_dataset,
    process_image,
    recreate_image,
)
from models import load_pretrained_model


def get_integrated_gradients_attribution_with_prediction(
    model: nn.Module, image: np.ndarray
):
    torch.manual_seed(0)
    np.random.seed(0)

    transform_normalize = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform_normalize(image)
    input_tensor = input_tensor.unsqueeze(0)

    print("Performing forward pass...")
    model = model.eval()
    output = model(input_tensor)
    output = F.softmax(output, dim=1)
    _, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    prediction = pred_label_idx.item()

    print("Generating visualization...")
    tic = time.time()
    visualization = IntegratedGradients(model)
    attributions = visualization.attribute(
        input_tensor, target=pred_label_idx, n_steps=5
    )
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))
    toc = time.time()
    print("Took %.02fs" % (toc - tic))

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom blue", [(0, "#ffffff"), (0.25, "#000000"), (1, "#000000")], N=256
    )

    fig, ax = viz.visualize_image_attr(
        attributions,
        method="heat_map",
        sign="positive",
        cmap=custom_cmap,
        show_colorbar=True,
    )

    ax.margins(0)
    fig.tight_layout(pad=0)

    return fig, prediction


def main():
    # Initialize the job, which will
    # load and verify all input parameters
    Job.initialize()

    vis_name = Job.function
    vis_options = dict(Job.options)
    assert vis_name == "IntegratedGradients"

    model_name = vis_options.pop("model")
    model = load_pretrained_model(model_name).eval()

    input_file_paths = list(map(lambda f: f.path, Job.input_files))
    input_dataset, is_temp_dataset = get_or_create_dataset(input_file_paths)

    output_dataset_path = Job.make_output_filepath(input_dataset.path.name)
    if os.path.exists(output_dataset_path):
        os.remove(output_dataset_path)
    output_dataset = build_output_figure_dataset(output_dataset_path)

    for item in input_dataset:
        input_image = np.float32(item.data)
        if input_dataset.metadata.channels_first:
            input_image = input_image.transpose([1, 2, 0])

        fig, prediction = get_integrated_gradients_attribution_with_prediction(
            model, input_image
        )

        arr = fig2arr(fig)
        output_dataset.add_item(
            name=item.name, data=arr, label=item.label, prediction=prediction
        )

    output_item = output_dataset[0]
    output_image = Image.fromarray(output_item.data)
    output_image_path = Job.make_output_filepath(output_item.name)
    output_image.save(output_image_path)

    labels_path = "data/imagenet_class_index.json"
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)
    output_label = idx_to_labels[str(output_item.prediction)][1]

    Job.add_output_file(str(output_dataset.path), is_extra=True)
    Job.add_output_file(
        output_image_path,
        is_modified=True,
        tags={"label": output_label, "mlsploit-visualize": "image"},
    )

    Job.commit_output()

    if is_temp_dataset:
        os.remove(input_dataset.path)


if __name__ == "__main__":
    main()
