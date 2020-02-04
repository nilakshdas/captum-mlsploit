import os
import json
import time

from mlsploit import Job

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

from models import load_pretrained_model


def get_integrated_gradients_attribution_with_prediction(model, image):
    torch.manual_seed(0)
    np.random.seed(0)

    labels_path = 'data/imagenet_class_index.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor()])
    transform_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])

    input_tensor = transform(image)
    input_tensor = transform_normalize(input_tensor)
    input_tensor = input_tensor.unsqueeze(0)

    print('Performing forward pass...')
    model = model.eval()
    output = model(input_tensor)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze_()
    predicted_label = idx_to_labels[str(pred_label_idx.item())][1]
    print('Predicted label: %s' % predicted_label)

    print('Generating visualization...')
    tic = time.time()
    visualization = IntegratedGradients(model)
    attributions = visualization.attribute(
        input_tensor, target=pred_label_idx, n_steps=50)
    attributions = attributions.squeeze().cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))
    toc = time.time()
    print('Took %.02fs' % (toc - tic))

    custom_cmap = LinearSegmentedColormap.from_list(
        'custom blue',
        [(0, '#ffffff'),
         (0.25, '#000000'),
         (1, '#000000')],
        N=256)

    fig, _ = viz.visualize_image_attr(
        attributions,
        method='heat_map', sign='positive',
        cmap=custom_cmap, show_colorbar=True)

    return fig, predicted_label


def main():
    # Initialize the job, which will
    # load and verify all input parameters
    Job.initialize()

    vis_name = Job.function
    vis_options = dict(Job.options)

    model_name = vis_options.pop('model')
    model = load_pretrained_model(model_name).eval()

    input_file_paths = list(map(lambda f: f.path, Job.input_files))
    input_file_path = input_file_paths[0]
    input_file_name = os.path.basename(input_file_path)

    original_image = Image.open(input_file_path)

    assert vis_name == 'IntegratedGradients'
    fig, label = get_integrated_gradients_attribution_with_prediction(
        model, original_image)

    output_file_path = Job.make_output_filepath(input_file_name)
    fig.savefig(output_file_path)

    Job.add_output_file(
        output_file_path, is_modified=True,
        tags={'label': label, 'mlsploit-visualize': 'image'})

    Job.commit_output()


if __name__ == '__main__':
    main()
